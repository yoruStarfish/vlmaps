import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Set

from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig
import torch
import gdown
import open3d as o3d
from scipy.spatial.transform import Rotation as R


from vlmaps.utils.lseg_utils import get_lseg_feat
from vlmaps.utils.mapping_utils import (
    load_3d_map,
    save_3d_map,
    cvt_pose_vec2tf,
    load_depth_npy,
    depth2pc,
    transform_pc,
    base_pos2grid_id_3d,
    project_point,
    get_sim_cam_mat,
)
from vlmaps.lseg.modules.models.lseg_net import LSegEncNet


def visualize_pc(pc: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])


class VLMapBuilder:
    def __init__(
        self,
        data_dir: Path,
        map_config: DictConfig,
        pose_path: Path,
        rgb_paths: List[Path],
        depth_paths: List[Path],
        base2cam_tf: np.ndarray,
        base_transform: np.ndarray,
    ):
        self.data_dir = data_dir
        self.pose_path = pose_path
        # self.rgb_paths = rgb_paths
        # self.depth_paths = depth_paths
        self.rgb_paths = sorted(rgb_paths, key=lambda x: int(x.stem))
        self.depth_paths = sorted(depth_paths, key=lambda x: int(x.stem))
        self.map_config = map_config
        self.base2cam_tf = base2cam_tf
        self.base_transform = base_transform
        self.rot_type = map_config.pose_info.rot_type

    def create_mobile_base_map(self):
        """
        build the 3D map centering at the first base frame
        """
        # access config info
        camera_height = self.map_config.pose_info.camera_height
        cs = self.map_config.cell_size
        gs = self.map_config.grid_size
        depth_sample_rate = self.map_config.depth_sample_rate

        self.base_poses = np.loadtxt(self.pose_path)
        if self.rot_type == "quat":
            self.init_base_tf = cvt_pose_vec2tf(self.base_poses[0])
        elif self.rot_type == "mat":
            self.init_base_tf = self.base_poses[0].reshape((4, 4))
        else:
            raise ValueError("Invalid rotation type")
        self.init_base_tf = self.base_transform @ self.init_base_tf @ np.linalg.inv(self.base_transform)
        self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
        self.init_cam_tf = self.init_base_tf @ self.base2cam_tf
        self.inv_init_cam_tf = np.linalg.inv(self.init_cam_tf)

        self.map_save_dir = self.data_dir / "vlmap"
        os.makedirs(self.map_save_dir, exist_ok=True)
        self.map_save_path = self.map_save_dir / "vlmaps.h5df"

        # init lseg model
        lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std = self._init_lseg()

        # init the map
        (
            vh,
            grid_feat,
            grid_pos,
            weight,
            occupied_ids,
            grid_rgb,
            mapped_iter_set,
            max_id,
        ) = self._init_map(camera_height, cs, gs, self.map_save_path)

        # load camera calib matrix in config
        calib_mat = np.array(self.map_config.cam_calib_mat).reshape((3, 3))
        cv_map = np.zeros((gs, gs, 3), dtype=np.uint8)
        height_map = -100 * np.ones((gs, gs), dtype=np.float32)

        color_map = np.ones((gs, gs, 3), dtype=np.uint8) * 255  # 白色背景
        print("🗺️ 開始生成彩色鳥瞰圖...")
        # 終極時空與旋轉對齊檢查
        print("\n" + "🔥 終極海膽殺手檢查 🔥")
        coords = self.base_poses[:, :3]
        x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
        y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])
        print(f"👉 軌跡 X 軸最大跨度: {x_range:.2f} 公尺")
        print(f"👉 軌跡 Y 軸最大跨度: {y_range:.2f} 公尺")
        
        # 檢查旋轉 (取出最後一幀的四元數並觀察)
        last_q = self.base_poses[-1, 3:7]
        print(f"👉 終點四元數 (q1, q2, q3, q4): {last_q}")
        print("="*50 + "\n")

        pbar = tqdm(zip(self.rgb_paths, self.depth_paths, self.base_poses), total=len(self.rgb_paths))
        for frame_i, (rgb_path, depth_path, base_posevec) in enumerate(pbar):
            # load data
            if self.rot_type == "quat":
                habitat_base_pose = cvt_pose_vec2tf(base_posevec)
            elif self.rot_type == "mat":
                habitat_base_pose = base_posevec.reshape((4, 4))
            else:
                raise ValueError("Invalid rotation type")
            
            base_pose = self.base_transform @ habitat_base_pose @ np.linalg.inv(self.base_transform)
            tf = self.inv_init_base_tf @ base_pose
            print(f"DEBUG: 機器人相對於原點的相對位移 [X, Y, Z] = {tf[0, 3]:.2f}, {tf[1, 3]:.2f}, {tf[2, 3]:.2f}")

            # theta = np.arctan2(tf[1, 0], tf[0, 0])
            # theta_deg = np.rad2deg(theta)
            # row, col, _ = base_pos2grid_id_3d(gs, cs, tf[0, 3], tf[1, 3], tf[2, 3])
            # trow, tcol, _ = base_pos2grid_id_3d(gs, cs, tf[0, 3] + tf[0, 0], tf[1, 3] + tf[1, 0], tf[2, 3])

            # cv2.circle(topdown, (col, row), 3, (0, 0, 255), -1)
            # cv2.circle(topdown, (tcol, trow), 3, (0, 255, 0), -1)

            bgr = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = load_depth_npy(depth_path)

            # # get pixel-aligned LSeg features
            pix_feats = get_lseg_feat(
                lseg_model, rgb, ["example"], lseg_transform, self.device, crop_size, base_size, norm_mean, norm_std
            )
            pix_feats_intr = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])

            # backproject depth point cloud
            # pc = self._backproject_depth(depth, calib_mat, depth_sample_rate, min_depth=0.1, max_depth=6)
            pc = self._backproject_depth(depth, calib_mat, depth_sample_rate, min_depth=0.5, max_depth=2)
            # 找到第 155 行附近
            # print(f"DEBUG: 原始點雲局部Z軸範圍: {pc[2].min():.2f} ~ {pc[2].max():.2f}")

            # 改成這樣防護：
            if pc.size == 0 or pc.shape[1] == 0:  # 依據你的 pc 形狀 (3, N) 或 (N, 3) 判斷
                print("WARNING: 當前影格生成的點雲為空！跳過此影格。")
                continue  # 或 return，視迴圈結構而定
            else:
                print(f"DEBUG: 原始點雲局部Z軸範圍: {pc[2].min():.2f} ~ {pc[2].max():.2f}")

            
            pc_transform = tf @ self.base_transform @ self.base2cam_tf
            pc_global = transform_pc(pc, pc_transform)  # (3, N)
            valid_mask = pc_global[2, :] > 0.0
            pc_global = pc_global[:, valid_mask]
            pc = pc[:, valid_mask]  # <-- 關鍵！讓 p_local 也能對齊

            # 1. 儲存「完全未經任何矩陣污染」的最原始點雲
            if frame_i == 0:
                print("\n" + "="*50)
                print("📸 正在生成完美帶色彩的單幀點雲...")
                # 準備乾淨的深度與影像
                clean_depth = np.nan_to_num(depth, nan=0.0)
                h, w = clean_depth.shape
                
                rgb_o3d = o3d.geometry.Image(rgb)
                depth_o3d = o3d.geometry.Image(clean_depth)
                
                # 融合 RGB 與 Depth
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False
                )
                
                # 建立相機矩陣並生成彩色點雲
                intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, calib_mat[0, 0], calib_mat[1, 1], calib_mat[0, 2], calib_mat[1, 2])
                colored_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
                
                # 小貼心：Open3D 預設投影出來的 Y 軸是朝下的，為了方便觀看我們把它翻轉過來
                colored_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                
                o3d.io.write_point_cloud("debug_0_ultimate_colored.pcd", colored_pcd)
                print("✅ 成功！請開啟 debug_0_ultimate_colored.pcd 看看！")
                print("="*50 + "\n")

            if frame_i == 1:
                print("\n" + "🔥 正在檢查海膽元兇 🔥")
                print("1. 你讀進來的第一筆軌跡資料 (Raw Pose):")
                print(base_posevec)
                
                # 檢查旋轉矩陣是否變成了「放大鏡」
                det = np.linalg.det(tf[:3, :3])
                print(f"2. 旋轉矩陣的行列式值 (純旋轉必須非常接近 1.0): {det:.6f}")
                print("="*50 + "\n")


            for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
                row, col, height = base_pos2grid_id_3d(gs, cs, p[0], p[1], p[2])
                if self._out_of_range(row, col, height, gs, vh):
                    continue

                px, py, pz = project_point(calib_mat, p_local)
                rgb_v = rgb[py, px, :]
                px, py, pz = project_point(pix_feats_intr, p_local)

#----------
                if 0 <= px < rgb.shape[1] and 0 <= py < rgb.shape[0]:                    
                    # 畫到彩色地圖上 (Z-Buffer 檢查：只顯示最高的點)
                    if 0 <= row < gs and 0 <= col < gs:
                        if p[2] > height_map[row, col]:
                            height_map[row, col] = p[2]
                            color_map[row, col] = rgb_v
#---------
                if height > height_map[row, col]:
                    height_map[row, col] = height
                    cv_map[row, col, :] = rgb_v

                # when the max_id exceeds the reserved size,
                # double the grid_feat, grid_pos, weight, grid_rgb lengths
                if max_id >= grid_feat.shape[0]:
                    # self._reserve_map_space(grid_feat, grid_pos, weight, grid_rgb)
                    grid_feat, grid_pos, weight, grid_rgb = self._reserve_map_space(grid_feat, grid_pos, weight, grid_rgb)

                # apply the distance weighting according to
                # ConceptFusion https://arxiv.org/pdf/2302.07241.pdf Sec. 4.1, Feature fusion
                radial_dist_sq = np.sum(np.square(p_local))
                sigma_sq = 0.6
                alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))

                # update map features
                if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                    feat = pix_feats[0, :, py, px]
                    occupied_id = occupied_ids[row, col, height]
                    if occupied_id == -1:
                        occupied_ids[row, col, height] = max_id
                        grid_feat[max_id] = feat.flatten() * alpha
                        grid_rgb[max_id] = rgb_v
                        weight[max_id] += alpha
                        grid_pos[max_id] = [row, col, height]
                        max_id += 1
                    else:
                        grid_feat[occupied_id] = (
                            grid_feat[occupied_id] * weight[occupied_id] + feat.flatten() * alpha
                        ) / (weight[occupied_id] + alpha)
                        grid_rgb[occupied_id] = (grid_rgb[occupied_id] * weight[occupied_id] + rgb_v * alpha) / (
                            weight[occupied_id] + alpha
                        )
                        weight[occupied_id] += alpha

            mapped_iter_set.add(frame_i)
            if frame_i % 100 == 99:
                print(f"Temporarily saving {max_id} features at iter {frame_i}...")
                self._save_3d_map(grid_feat, grid_pos, weight, grid_rgb, occupied_ids, mapped_iter_set, max_id)
        color_map_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
        cv2.imwrite("debug_ultimate_color_map.png", color_map_bgr)
        print("✅ 高畫質彩色鳥瞰圖已生成：debug_ultimate_color_map.png")

        # ▼▼▼ 請在這裡插入新的匯出 3D 彩色點雲程式碼 ▼▼▼
        print("正在匯出 3D 彩色全局點雲 (Global Colored Point Cloud)...")
        # 1. 取得真正有被探索到的有效網格 (只取 0 ~ max_id 的部分)
        valid_grid_pos = grid_pos[:max_id]
        valid_grid_rgb = grid_rgb[:max_id]

        # 2. 將網格座標轉換回真實世界的物理座標 (公尺)
        # 這裡的 cs 是你 config 中的 cell_size (例如 0.05)
        X = valid_grid_pos[:, 1].astype(float) * cs
        Y = valid_grid_pos[:, 0].astype(float) * cs
        Z = valid_grid_pos[:, 2].astype(float) * cs
        real_points = np.column_stack((X, Y, Z))

        # 3. 處理顏色資訊：Open3D 要求 RGB 數值必須落在 0.0 ~ 1.0 之間
        # 原始的 grid_rgb 是 0~255 的 uint8，所以我們除以 255.0
        real_colors = valid_grid_rgb.astype(float) / 255.0

        # 4. 建立 Open3D 點雲物件並塞入座標與顏色
        global_color_pcd = o3d.geometry.PointCloud()
        global_color_pcd.points = o3d.utility.Vector3dVector(real_points)
        global_color_pcd.colors = o3d.utility.Vector3dVector(real_colors) # 這行是上色關鍵！

        # 5. 將點雲輸出成 PCD 檔案
        pcd_filename = "debug_full_global_colored_map.pcd"
        o3d.io.write_point_cloud(pcd_filename, global_color_pcd)
        print(f"✅ 3D 彩色全局點雲已匯出：{pcd_filename}")
        # ▲▲▲ 插入結束 ▲▲▲
            
        self._save_3d_map(grid_feat, grid_pos, weight, grid_rgb, occupied_ids, mapped_iter_set, max_id)

    def create_camera_map(self):
        """
        TODO: To be implemented
        build the 3D map centering at the first camera frame. We require that the camera is initialized
        horizontally (the optical axis is parallel to the floor at the first frame).
        """
        return NotImplementedError

    def _init_map(self, map_height: float, cs: float, gs: int, map_path: Path) -> Tuple:
        """
        initialize a voxel grid of size (gs, gs, vh), vh = map_height / cs, each voxel is of
        size cs
        """
        # init the map related variables
        vh = 40 #int(map_height / cs)
        grid_feat = np.zeros((gs * gs, self.clip_feat_dim), dtype=np.float32)
        grid_pos = np.zeros((gs * gs, 3), dtype=np.int32)
        occupied_ids = -1 * np.ones((gs, gs, vh), dtype=np.int32)
        weight = np.zeros((gs * gs), dtype=np.float32)
        grid_rgb = np.zeros((gs * gs, 3), dtype=np.uint8)
        mapped_iter_set = set()
        mapped_iter_list = list(mapped_iter_set)
        max_id = 0

        # check if there is already saved map
        if os.path.exists(map_path):
            (
                mapped_iter_list,
                grid_feat,
                grid_pos,
                weight,
                occupied_ids,
                grid_rgb,
            ) = load_3d_map(self.map_save_path)
            mapped_iter_set = set(mapped_iter_list)
            max_id = grid_feat.shape[0]

        return vh, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, mapped_iter_set, max_id

    def _init_lseg(self):
        crop_size = 480  # 480
        base_size = 520  # 520
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
        model_state_dict = lseg_model.state_dict()
        checkpoint_dir = Path(__file__).resolve().parents[1] / "lseg" / "checkpoints"
        checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
        os.makedirs(checkpoint_dir, exist_ok=True)
        if not checkpoint_path.exists():
            print("Downloading LSeg checkpoint...")
            # the checkpoint is from official LSeg github repo
            # https://github.com/isl-org/lang-seg
            checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
            gdown.download(checkpoint_url, output=str(checkpoint_path))

        # pretrained_state_dict = torch.load(checkpoint_path, map_location=self.device) # in python 3.8
        pretrained_state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
        model_state_dict.update(pretrained_state_dict)
        lseg_model.load_state_dict(pretrained_state_dict)

        lseg_model.eval()
        lseg_model = lseg_model.to(self.device)

        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        lseg_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.clip_feat_dim = lseg_model.out_c
        return lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std

    def _backproject_depth(
        self,
        depth: np.ndarray,
        calib_mat: np.ndarray,
        depth_sample_rate: int,
        min_depth: float = 0.1,
        max_depth: float = 10,
    ) -> np.ndarray:
        pc, mask = depth2pc(depth, intr_mat=calib_mat, min_depth=min_depth, max_depth=max_depth)  # (3, N)
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]

        # # --- 在 return pc 之前加入這段，殺死邊緣幽靈點 ---
        # if pc.shape[1] > 50:
        #     pcd_filter = o3d.geometry.PointCloud()
        #     pcd_filter.points = o3d.utility.Vector3dVector(pc.T)
            
        #     # 尋找周圍 20 個點，標準差閾值設為 1.5，將離群點全部刪除
        #     clean_pcd, ind = pcd_filter.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        #     pc = np.asarray(clean_pcd.points).T
        # # ---------------------------------------------------
        # --- 📍 終極點雲優化組合拳 📍 ---
        # 1. 將 numpy 點轉為 Open3D PCD 格式以便濾波
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.T)
        
        # 2. 招式一： voxel_downsample (體素降採樣)
        # 這能把不同 FPS 拍到的點雲進行均勻化，讓地圖點雲密度一致，並能初步平滑表面。
        # 此處使用 voxel_size=0.01 (1公分)。過大的 voxel 會吃掉桌腳，請小心調整。
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        if len(pcd.points) > 50:
            # 3. 招式二： remove_radius_outlier (半徑濾波)
            # 這是RealSense幽靈點的終極殺手。比起統計濾波，它能更精準地砍掉周圍沒有邻居陪伴的孤立「毛邊點」。
            # nb_points=8: 半徑內點數少於8個就砍。
            # radius=0.05: 半徑為5公分。這兩個參數配合能很好的剃除桌角和牆角的飛行像素。
            pcd, ind = pcd.remove_radius_outlier(nb_points=8, radius=0.05)
            
            # 4. (可選) 招式三：remove_statistical_outlier (統計濾波)
            # 如果加上半徑濾波後邊緣還是不夠利，可以把這行也加上去進行最後的精細清理。
            # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
            
        # 5. 將乾淨的點雲轉回原本的 numpy 格式並轉置 back to (3, N)
        pc = np.asarray(pcd.points).T
        # --------------------------------------------------
        
        return pc
        return pc

    # def _backproject_depth(
    #     self,
    #     depth: np.ndarray,
    #     calib_mat: np.ndarray,
    #     depth_sample_rate: int,
    #     min_depth: float = 0.1,
    #     max_depth: float = 10,
    # ) -> np.ndarray:
    #     h, w = depth.shape
    #     fx, fy = calib_mat[0, 0], calib_mat[1, 1]
    #     cx, cy = calib_mat[0, 2], calib_mat[1, 2]
        
    #     # 🚨 關鍵修復：把 NaN 換回 0.0！不然 Open3D 會拉出放射狀射線
    #     clean_depth = np.nan_to_num(depth, nan=0.0)
        
    #     depth_o3d = o3d.geometry.Image(clean_depth)
    #     intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        
    #     pcd = o3d.geometry.PointCloud.create_from_depth_image(
    #         depth_o3d, intrinsic, depth_scale=1.0, depth_trunc=max_depth
    #     )
        
    #     pc = np.asarray(pcd.points).T
        
    #     valid_mask = pc[2, :] > min_depth
    #     pc = pc[:, valid_mask]
        
    #     if depth_sample_rate > 1:
    #         shuffle_mask = np.arange(pc.shape[1])
    #         np.random.shuffle(shuffle_mask)
    #         shuffle_mask = shuffle_mask[::depth_sample_rate]
    #         pc = pc[:, shuffle_mask]
            
    #     return pc
    
    def _out_of_range(self, row: int, col: int, height: int, gs: int, vh: int) -> bool:
        return col >= gs or row >= gs or height >= vh or col < 0 or row < 0 or height < 0

    def _reserve_map_space(
        self, grid_feat: np.ndarray, grid_pos: np.ndarray, weight: np.ndarray, grid_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        grid_feat = np.concatenate(
            [
                grid_feat,
                np.zeros((grid_feat.shape[0], grid_feat.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        grid_pos = np.concatenate(
            [
                grid_pos,
                np.zeros((grid_pos.shape[0], grid_pos.shape[1]), dtype=np.int32),
            ],
            axis=0,
        )
        weight = np.concatenate([weight, np.zeros((weight.shape[0]), dtype=np.int32)], axis=0)
        grid_rgb = np.concatenate(
            [
                grid_rgb,
                np.zeros((grid_rgb.shape[0], grid_rgb.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        return grid_feat, grid_pos, weight, grid_rgb

    def _save_3d_map(
        self,
        grid_feat: np.ndarray,
        grid_pos: np.ndarray,
        weight: np.ndarray,
        grid_rgb: np.ndarray,
        occupied_ids: Set,
        mapped_iter_set: Set,
        max_id: int,
    ) -> None:
        grid_feat = grid_feat[:max_id]
        grid_pos = grid_pos[:max_id]
        weight = weight[:max_id]
        grid_rgb = grid_rgb[:max_id]
        save_3d_map(self.map_save_path, grid_feat, grid_pos, weight, occupied_ids, list(mapped_iter_set), grid_rgb)
