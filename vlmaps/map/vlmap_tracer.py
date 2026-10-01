from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple, Union
import gdown

from tqdm import tqdm
import clip
import cv2
import torchvision.transforms as transforms
import numpy as np
import open3d as o3d

# --- 加入以下三行來修復 timm 與 numpy 的 allow_pickle 衝突 ---
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
# ---------------------------------------------------------

from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
import torch

base_dir = os.path.expanduser('~/vlmaps_ws')
vlmaps_src_dir = os.path.join(base_dir, 'vlmapsSrc')



# 將兩個層級的路徑都加入，確保不會再發生 ModuleNotFoundError
if vlmaps_src_dir not in sys.path:
    sys.path.insert(0, vlmaps_src_dir)


# import vlmaps
# print(vlmaps)
# print(vlmaps.__file__)
from vlmaps.utils.clip_utils import get_text_feats_multiple_templates
from vlmaps.utils.visualize_utils import pool_3d_label_to_2d

# from utils.ai2thor_constant import ai2thor_class_list
# from utils.clip_mapping_utils import load_map
# from utils.planning_utils import (
#     find_similar_category_id,
#     get_dynamic_obstacles_map,
#     get_lseg_score,
#     get_segment_islands_pos,
#     mp3dcat,
#     segment_lseg_map,
# )
from vlmaps.map.vlmap_builder import VLMapBuilder
from vlmaps.map.vlmap_builder_cam import VLMapBuilderCam
from vlmaps.utils.mapping_utils import load_3d_map
from vlmaps.map.map import Map
from vlmaps.utils.index_utils import find_similar_category_id, get_segment_islands_pos, get_dynamic_obstacles_map_3d
from vlmaps.utils.clip_utils import get_lseg_score


class VLMapTracer(Map):
    def __init__(self, map_config: DictConfig, data_dir: str = ""):
        super().__init__(map_config, data_dir=data_dir)
        self.scores_mat = None
        self.categories = None

    def create_map(self, data_dir: Union[Path, str]) -> None:
        print(f"Creating map for scene at: ", data_dir)
        self._setup_paths(data_dir)
        if self.map_config.pose_info.pose_type == "mobile_base":
            self.map_builder = VLMapBuilder(
                self.data_dir,
                self.map_config,
                self.pose_path,
                self.rgb_paths,
                self.depth_paths,
                self.base2cam_tf,
                self.base_transform,
            )
            self.map_builder.create_mobile_base_map()
                
        elif self.map_config.pose_info.pose_type == "camera_base":
            self.map_builder = VLMapBuilderCam(
                self.data_dir,
                self.map_config,
                self.pose_path,
                self.rgb_paths,
                self.depth_paths,
                self.base2cam_tf,
                self.base_transform,
            )
            self.map_builder.create_camera_map()
            o3d.io.write_point_cloud("check_raw_pcd.pcd", self.map_builder.global_pcd)
        else:
            raise ValueError("Invalid pose type")

    def load_map(self, data_dir: str) -> bool:
        self._setup_paths(data_dir)
        print(self.data_dir)
        if self.map_config.pose_info.pose_type == "mobile_base":
            self.map_save_path = Path(data_dir) / "vlmap" / "vlmaps.h5df"
            print(self.map_save_path)
            if not self.map_save_path.exists():
                assert False, "Loading VLMap failed because the file doesn't exist."
            (
                self.mapped_iter_list,
                self.grid_feat,
                self.grid_pos,
                self.weight,
                self.occupied_ids,
                self.grid_rgb,
            ) = load_3d_map(self.map_save_path)
        elif self.map_config.pose_info.pose_type == "camera_base":
            self.map_save_path = Path(data_dir) / "vlmap_cam" / "vlmaps_cam.h5df"
            print(self.map_save_path)
            if not self.map_save_path.exists():
                assert False, "Loading VLMap failed because the file doesn't exist."
            (
                self.mapped_iter_list,
                self.grid_feat,
                self.grid_pos,
                self.weight,
                self.occupied_ids,
                self.grid_rgb,
                self.pcd_min,
                self.pcd_max,
                self.cs,
            ) = VLMapBuilderCam.load_3d_map(self.map_save_path)
        else:
            raise ValueError("Invalid pose type")

        return True

    def _init_clip(self, clip_version="ViT-B/32"):
        if hasattr(self, "clip_model"):
            print("clip model is already initialized")
            return
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.clip_version = clip_version
        self.clip_feat_dim = {
            "RN50": 1024,
            "RN101": 512,
            "RN50x4": 640,
            "RN50x16": 768,
            "RN50x64": 1024,
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
        }[self.clip_version]
        print("Loading CLIP model...")
        self.clip_model, self.preprocess = clip.load(self.clip_version)  # clip.available_models()
        self.clip_model.to(self.device).eval()

    def init_categories(self, categories: List[str]) -> np.ndarray:
        self.categories = categories
        self.scores_mat = get_lseg_score(
            self.clip_model,
            self.categories,
            self.grid_feat,
            self.clip_feat_dim,
            use_multiple_templates=True,
            add_other=True,
        )  # score for name and other
        return self.scores_mat

    def index_map(self, language_desc: str, with_init_cat: bool = True):
        if with_init_cat and self.scores_mat is not None and self.categories is not None:
            cat_id = find_similar_category_id(language_desc, self.categories)
            scores_mat = self.scores_mat
        else:
            if with_init_cat:
                raise Exception(
                    "Categories are not preloaded. Call init_categories(categories: List[str]) to initialize categories."
                )
            scores_mat = get_lseg_score(
                self.clip_model,
                [language_desc],
                self.grid_feat,
                self.clip_feat_dim,
                use_multiple_templates=True,
                add_other=True,
            )  # score for name and other
            cat_id = 0

        max_ids = np.argmax(scores_mat, axis=1)
        mask = max_ids == cat_id
        return mask

    def customize_obstacle_map(
        self,
        potential_obstacle_names: List[str],
        obstacle_names: List[str],
        vis: bool = False,
    ):
        if self.obstacles_cropped is None and self.obstacles_map is None:
            self.generate_obstacle_map(h_min=0.35, h_max=1.5)
        if not hasattr(self, "clip_model"):
            print("init_clip in customize obstacle map")
            self._init_clip()

        self.obstacles_new_cropped = get_dynamic_obstacles_map_3d(
            self.clip_model,
            self.obstacles_cropped,
            self.map_config.potential_obstacle_names,
            self.map_config.obstacle_names,
            self.grid_feat,
            self.grid_pos,
            self.rmin,
            self.cmin,
            self.clip_feat_dim,
            vis=vis,
        )
        self.obstacles_new_cropped = Map._dilate_map(
            self.obstacles_new_cropped == 0,
            self.map_config.dilate_iter,
            self.map_config.gaussian_sigma,
        )
        self.obstacles_new_cropped = self.obstacles_new_cropped == 0

        obs_indices = np.argwhere(self.obstacles_new_cropped == 0) # 0 means occupied, 1 means free
      


    def get_pos(self, name: str) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], Any]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        assert self.categories
        # cat_id = find_similar_category_id(name, self.categories)
        # labeled_map_cropped = self.scores_mat.copy()  # (N, C) N: number of voxels, C: number of categories
        # labeled_map_cropped = np.argmax(labeled_map_cropped, axis=1)  # (N,)
        # pc_mask = labeled_map_cropped == cat_id # (N,)
        # self.grid_pos[pc_mask]
        pc_mask = self.index_map(name, with_init_cat=True)
        mask_2d = pool_3d_label_to_2d(pc_mask, self.grid_pos, self.gs)
        mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
        # print(f"showing mask for object cat {name}")
        # cv2.imshow(f"mask_{name}", (mask_2d.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey()

        # --- 🚨 新增防彈衣：在進入死亡濾鏡前，先強制膨脹一次！ ---
        # 把 1 像素的微弱訊號，擴大成一個九宮格的實體區塊
        from scipy.ndimage import binary_dilation # 確保有引入
        protected_mask = binary_dilation(mask_2d > 0, iterations=1)
        # ----------------------------------------------------

        foreground = binary_closing(protected_mask, iterations=3)
        foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
        foreground = foreground > 0.5
        # cv2.imshow(f"mask_{name}_gaussian", (foreground * 255).astype(np.uint8))
        foreground = binary_dilation(foreground)
        # cv2.imshow(f"mask_{name}_processed", (foreground.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey()

        contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)
        # print("centers", centers)

        # whole map position
        for i in range(len(contours)):
            centers[i][0] += self.rmin
            centers[i][1] += self.cmin
            bbox_list[i][0] += self.rmin
            bbox_list[i][1] += self.rmin
            bbox_list[i][2] += self.cmin
            bbox_list[i][3] += self.cmin
            for j in range(len(contours[i])):
                contours[i][j, 0] += self.rmin
                contours[i][j, 1] += self.cmin

        return contours, centers, bbox_list
    
    def calculate_real_world_coordinates(self):
        """
        Convert grid indices to real-world coordinates in meters.
        :param grid_indices: An array of shape (N, 3) where each row is (row_index, col_index, height_index).
        :return: Map center with real-world coordinates in meters.
        """
        if not hasattr(self, 'cs'):
            raise AttributeError("Cell size (cs) is not defined. Ensure the map is loaded and cs is set.")
        
        # Convert grid indices to real-world coordinates
        print("self.cmin: ", self.cmin)
        print("self.rmax: ", self.rmax)
        OriginX = (self.cmin - self.gs / 2) * self.cs
        OriginY = - (self.rmax - self.gs / 2) * self.cs
        return OriginX, OriginY

    
if __name__ == "__main__":
    # Example usage
    map_config = OmegaConf.load("vlmapsSrc/config/map_config/vlmaps_tracer.yaml")
    data_dir = os.path.join(base_dir, "dataset")
    vlmap_tracer = VLMapTracer(map_config, data_dir)
    
    # 1. 建圖階段 (有 .h5df 後即可註解掉，節省時間)
    # vlmap_tracer.create_map(data_dir)
    
    # 2. 載入地圖
    vlmap_tracer.load_map(data_dir)
    
    # 3. 產生與自定義障礙物地圖 (修復 Bug 1, 4)
    # 設定 h_min=0.15 過濾掉 15 公分以下的地板雜訊
    vlmap_tracer.generate_obstacle_map(h_min=0.35, h_max=1.5)
    # 補上缺少的參數
    vlmap_tracer.customize_obstacle_map(
        potential_obstacle_names=map_config.potential_obstacle_names,
        obstacle_names=map_config.obstacle_names
    )
    cv2.imwrite("debug_obstacle_map.png", (vlmap_tracer.obstacles_new_cropped.astype(np.uint8) * 255))
    
    # 4. 初始化 AI 模型與搜尋清單
    vlmap_tracer._init_clip()
    categories = [
        "chair",
        "table",
        "wall",
        "cardboard box",
        "sandwich",
        "paper cup",
        "kettle",
        "bottle",
        "floor",
        "other"
    ]
    vlmap_tracer.init_categories(categories=categories)
    
    # 5. 執行搜尋
    target = "sandwich"  # 你要找的目標物品
    mask_1d = vlmap_tracer.index_map(target)  # 取得 1D 點雲遮罩
    contours, centers, bbox_list = vlmap_tracer.get_pos(target)

    # ---------------------------------------------------------
    # 終極 Debug 與視覺化輸出區塊
    # ---------------------------------------------------------
    
    # 1. 文字輸出 (Print Centers)
    print(f"-> 總共找到了 {len(centers)} 個 {target}！")
    for i, center in enumerate(centers):
        print(f"   [{i+1}] 中心點座標: Y(Row)={int(center[0])}, X(Col)={int(center[1])}")

    # 2. 儲存原始 Mask (修復 Bug 3: 1D 轉 2D 裁切)
    # 將 1D 點雲特徵壓扁成 2D 網格
    mask_2d = pool_3d_label_to_2d(mask_1d, vlmap_tracer.grid_pos, vlmap_tracer.gs)
    # 裁切掉沒有探索過的黑色邊界
    mask_2d_cropped = mask_2d[vlmap_tracer.rmin : vlmap_tracer.rmax + 1, vlmap_tracer.cmin : vlmap_tracer.cmax + 1]
    
    mask_img = (mask_2d_cropped.astype(np.uint8) * 255)
    cv2.imwrite(f"debug_1_raw_mask_{target}.png", mask_img)
    print(f"✅ 原始遮罩已儲存: debug_1_raw_mask_{target}.png")

    # 3. 製作像素級語意分割疊加圖 (Segmentation Overlay)
    # 準備底圖：把黑白的障礙物地圖轉成彩色 3 通道
    obs_map_img = (vlmap_tracer.obstacles_new_cropped.astype(np.uint8) * 255)
    vis_img = cv2.cvtColor(obs_map_img, cv2.COLOR_GRAY2BGR)
    
    # 複製一張用來做半透明疊加的圖層
    overlay = vis_img.copy()
    
    # 定義 table 的專屬顏色，例如紅色 (BGR 格式: [0, 0, 255])
    # 你可以依據不同類別換顏色，例如椅子用綠色 [0, 255, 0]
    target_color = [0, 0, 255] 
    
    # 找出 mask_2d_cropped 中被判定為 target (數值 > 0) 的所有像素位置
    target_pixels = mask_2d_cropped > 0
    
    # 將這些像素在 overlay 圖層上全部塗成紅色
    overlay[target_pixels] = target_color
    
    # 使用 cv2.addWeighted 進行半透明融合 (Alpha Blending)
    # 0.5 代表透明度為 50%，這樣你既能看到紅色色塊，也能隱約看到底下的桌腳輪廓
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
    
    h, w = vis_img.shape[:2]
    print(f"🔍 目前地圖的尺寸: 寬(X)={w}, 高(Y)={h}")
    
    print(f"🎯 準備畫目標 {target} 的中心點...")
    for i, center in enumerate(centers):
        # 1. 先取得在 1000x1000 全域地圖上的原始座標
        global_cy = int(center[0]) # Y
        global_cx = int(center[1]) # X
        
        # 2. 扣除裁切的起點，轉換回 191x229 小地圖上的相對座標
        # 注意：rmin 是列(Y)的起點，cmin 是欄(X)的起點
        cy = global_cy - vlmap_tracer.rmin
        cx = global_cx - vlmap_tracer.cmin
        
        print(f"   -> 圓點 {i+1} 原始全域座標: (X={global_cx}, Y={global_cy})")
        print(f"   -> 轉換為裁切畫布座標: (X={cx}, Y={cy})")
        
        # 3. 畫圖與檢查
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(vis_img, (cx, cy), 4, (0, 255, 255), -1) # 畫黃色實心
            print("      ✅ 畫圖成功！")
        else:
            print("      ❌ 警告：轉換後座標依然超出圖片範圍！")

        
    cv2.imwrite(f"debug_2_segmentation_overlay_{target}.png", vis_img)
    print(f"✅ 像素級語意分割圖已儲存: debug_2_segmentation_overlay_{target}.png")

    # 1. 取得所有點雲的網格座標 (N, 3)，格式為 [row, col, height]
    all_grids = vlmap_tracer.grid_pos

    # 2. 定義真實世界的高度過濾 (對應你的 PCD 視覺結果)
    # 假設你的 cs (cell_size) 是 0.05
    # height = 6 ~ 24 大約代表真實世界的 0.3m ~ 1.2m
    cs = vlmap_tracer.cs  # cell size
    H_MIN_METER = 0.35 
    H_MAX_METER = 1.5

    # 🚨 修正 Z 軸高度範圍 (將公尺轉為網格索引索引)
    h_min_idx = int(H_MIN_METER / cs) # indices = meters / cell_size [cite: 2]
    h_max_idx = int(H_MAX_METER / cs)

    # 3. 篩選出真正屬於「牆壁與桌子」的點
    valid_mask = (all_grids[:, 2] >= h_min_idx) & (all_grids[:, 2] <= h_max_idx)
    obstacle_grids = all_grids[valid_mask]
    print(f"總點數: {len(all_grids)}, 障礙物點數: {len(obstacle_grids)}")

    # 4. 建立一張純白色的 2D 畫布
    # 根據你的網格範圍，建立一張足夠大的地圖 (例如 1000x1000)
    map_size = 1000
    geometry_map = np.ones((map_size, map_size), dtype=np.uint8) * 255

    # 5. 把障礙物畫上去 (塗成黑色 0)
    unique_obstacles = set()
    for grid in obstacle_grids:
        unique_obstacles.add((int(grid[0]), int(grid[1])))

    # 現在對 unique_obstacles 進行操作，就不會因為軌跡重疊而塗滿整個地圖
    for row, col in unique_obstacles:
        if 0 <= row < map_size and 0 <= col < map_size:
            geometry_map[row, col] = 0

    # 6. 自動裁切掉外圍沒有去過的空白區域
    # 找到所有黑色像素的邊界
    coords = cv2.findNonZero(255 - geometry_map) 
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # 往外留 20 個像素的白邊
        geometry_map_cropped = geometry_map[max(0, y-20):y+h+20, max(0, x-20):x+w+20]
    else:
        geometry_map_cropped = geometry_map

    # 7. 儲存這張最真實的地圖
    cv2.imwrite("debug_99_pure_geometry_map.png", geometry_map_cropped)
    print("✅ 純幾何地圖已儲存: debug_99_pure_geometry_map.png")

    print("正在匯出完整的 3D 全局地圖...")
    # 你的 cs 是 0.1，我們把網格座標乘上 0.1，還原成真實世界的公尺 (meters)
    real_points = vlmap_tracer.grid_pos.astype(float) * 0.1

    global_pcd = o3d.geometry.PointCloud()
    global_pcd.points = o3d.utility.Vector3dVector(real_points)

    o3d.io.write_point_cloud("debug_full_global_map.pcd", global_pcd)
    print("✅ 完整全局 3D 地圖已匯出：debug_full_global_map.pcd")

    originX, originY = vlmap_tracer.calculate_real_world_coordinates()
    print(f"地圖原點 (Origin) 的真實世界座標: X={originX:.3f} m, Y={originY:.3f} m")
