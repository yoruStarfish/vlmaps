import sys
import os
import argparse
import math
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import clip
from omegaconf import OmegaConf

# --- 1. 路徑與模組搜尋防呆設定 ---
base_dir = os.path.expanduser('~/vlmaps_ws')
vlmaps_src_dir = os.path.join(base_dir, 'vlmapsSrc')
vlmaps_core_dir = os.path.join(vlmaps_src_dir, 'vlmaps')

# 將兩個層級的路徑都加入，確保不會再發生 ModuleNotFoundError
if vlmaps_src_dir not in sys.path:
    sys.path.append(vlmaps_src_dir)
if vlmaps_core_dir not in sys.path:
    sys.path.append(vlmaps_core_dir)

# --- 2. PyTorch Vision Transformer 權重下載修復 ---
import timm
backup_url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth'
if 'vit_large_patch16_384' in timm.models.vision_transformer.default_cfgs:
    timm.models.vision_transformer.default_cfgs['vit_large_patch16_384']['url'] = backup_url

# --- 3. 載入必要的 VLMaps 與 LSeg 模組 ---
from utils.mapping_utils import load_pose, save_map, depth2pc, transform_pc, get_sim_cam_mat, pos2grid_id, project_point
from lseg.modules.models.lseg_net import LSegEncNet
from lseg.additional_utils.models import resize_image, pad_image, crop_image
# 載入我們剛剛分析的官方 VLMap 類別
from vlmaps.map.vlmap import VLMap


# --- 4. 輔助函式保留 ---
def load_depth(depth_filepath):
    depth = cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000.0
    return depth

def get_lseg_feat(model: LSegEncNet, image: np.array, labels, transform, crop_size=480, base_size=520, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
    # 維持原本的特徵提取邏輯
    image = transform(image).unsqueeze(0).cuda()
    batch, _, h, w = image.size()
    stride_rate = 2.0/3.0
    stride = int(crop_size * stride_rate)

    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height

    cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})

    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        with torch.no_grad():
            outputs, logits = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)
    else:
        if short_size < crop_size:
            pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        else:
            pad_img = cur_img
        _,_,ph,pw = pad_img.shape
        h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
        with torch.cuda.device_of(image):
            with torch.no_grad():
                outputs = image.new().resize_(batch, model.out_c,ph,pw).zero_().cuda()
                count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)
                pad_crop_img = pad_image(crop_img, norm_mean, norm_std, crop_size)
                with torch.no_grad():
                    output, _ = model(pad_crop_img, labels)
                cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                outputs[:,:,h0:h1,w0:w1] += cropped
                count_norm[:,:,h0:h1,w0:w1] += 1
        outputs = outputs / count_norm
        outputs = outputs[:,:,:height,:width]
    outputs = outputs.cpu().numpy() 
    return outputs


# --- 5. 核心：繼承 VLMap 並覆寫建圖邏輯 ---
class TracerVLMap(VLMap):
    def __init__(self, data_dir: str, camera_height: float = 0.69, gs: int = 1000, cs: float = 0.05):
        # 偽造一個 DictConfig 以滿足官方 VLMap 父類別的要求
        mock_config = OmegaConf.create({
            "grid_size": gs,
            "cell_size": cs,
            "pose_info": {
                "pose_type": "mobile_base",
                "camera_height": camera_height,
                "base2cam_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 
                "base_forward_axis": [1.0, 0.0, 0.0],
                "base_left_axis": [0.0, 1.0, 0.0],
                "base_up_axis": [0.0, 0.0, 1.0]
            }
        })
        # 呼叫父類別初始化
        super().__init__(map_config=mock_config, data_dir=data_dir)
        
        self.camera_height = camera_height
        self.depth_sample_rate = 5
        self.clip_feat_dim = 512
        
        # 初始化建圖所需的陣列
        self.grid = np.zeros((self.gs, self.gs, self.clip_feat_dim), dtype=np.float32)
        self.weight = np.zeros((self.gs, self.gs), dtype=float)
        
        # 障礙物地圖初始化為 0 (可通行/未探索)。偵測到障礙物時設為 1。
        self.obstacles = np.zeros((self.gs, self.gs), dtype=np.uint8) 
        
        # 彩色鳥瞰圖所需
        self.color_top_down = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        self.color_top_down_height = -10.0 * np.ones((self.gs, self.gs), dtype=np.float32)

    # 覆寫 create_map，使用我們客製化的座標轉換與 Z-Buffer 邏輯
    def create_map(self):
        print(f"[系統] 開始使用 TracerVLMap 建立地圖，資料來源: {self.data_dir}")
        
        # 準備 LSeg 模型
        lang = "wall,chair,table,floor,other"
        labels = lang.split(",")
        crop_size = 480 
        base_size = 520 
        
        model = LSegEncNet(lang, arch_option=0, block_depth=0, activation='lrelu', crop_size=crop_size)
        checkpoint_path = os.path.join(vlmaps_core_dir, "lseg/checkpoints/demo_e200.ckpt") 
        pretrained_state_dict = torch.load(checkpoint_path)
        pretrained_state_dict = {k.lstrip('net.'): v for k, v in pretrained_state_dict['state_dict'].items()}
        model.load_state_dict(pretrained_state_dict, strict=False)
        model.eval()
        model = model.cuda()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # 讀取資料
        rgb_list = sorted([os.path.join(self.rgb_dir, x) for x in os.listdir(self.rgb_dir)])
        depth_list = sorted([os.path.join(self.depth_dir, x) for x in os.listdir(self.depth_dir)  if x.endswith(".png")])
        pose_file = os.path.join(self.data_dir, "poses.txt")
        pose_data = np.loadtxt(pose_file)

        tf_list = []
        data_iter = zip(rgb_list, depth_list)
        pbar = tqdm(data_iter, total=len(rgb_list), desc="建圖進度", ncols=100)

        for frame_idx, (rgb_path, depth_path) in enumerate(data_iter):
            bgr = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            from scipy.spatial.transform import Rotation as R

            x, y, z, qx, qy, qz, qw = pose_data[frame_idx]

            pose = np.eye(4)
            pose[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            pose[:3, 3] = [x, y, z]

            tf_list.append(pose)
            if len(tf_list) == 1: 
                init_tf_inv = np.linalg.inv(tf_list[0]) 

            tf = init_tf_inv @ pose 
            depth = load_depth(depth_path)
            pix_feats = get_lseg_feat(model, rgb, labels, transform, crop_size, base_size) 
            
            pc, mask = depth2pc(depth) 
            shuffle_mask = np.arange(pc.shape[1]) 
            np.random.shuffle(shuffle_mask)
            shuffle_mask = shuffle_mask[::self.depth_sample_rate] 
            mask = mask[shuffle_mask]
            pc = pc[:, shuffle_mask]
            pc = pc[:, mask]

            # 光學座標系轉換至機器人座標系 (Camera_link)
            pc_cl = np.zeros_like(pc)
            pc_cl[0, :] = pc[2, :]   
            pc_cl[1, :] = -pc[0, :]  
            pc_cl[2, :] = -pc[1, :]  

            # 轉換至 Map 座標系，確保 Z 軸為真實世界高度
            pc_global = transform_pc(pc_cl, tf) 

            rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])
            # 取得特徵圖的真實維度
            feat_h, feat_w = pix_feats.shape[2], pix_feats.shape[3]
            rgb_h, rgb_w = rgb.shape[0], rgb.shape[1]

            # 計算縮放比例
            scale_y = feat_h / rgb_h
            scale_x = feat_w / rgb_w

            # 建立原始的相機矩陣 (基於 RGB 影像尺寸)
            feat_cam_mat = get_sim_cam_mat(rgb_h, rgb_w)

            # 【關鍵修正】：將相機矩陣的內參（焦距與中心點）按照縮放比例調整
            feat_cam_mat[0, 0] *= scale_x # fx
            feat_cam_mat[1, 1] *= scale_y # fy
            feat_cam_mat[0, 2] *= scale_x # cx
            feat_cam_mat[1, 2] *= scale_y # cy
            debug_img = rgb.copy()
            for p, p_opt in zip(pc_global.T, pc.T): 
                x, y = pos2grid_id(self.gs, self.cs, p[0], p[1])

                if x >= self.gs or y >= self.gs or x < 0 or y < 0: 
                    continue

                # 高度過濾：略過天花板與地底雜訊
                if p[2] > 2.0 or p[2] < -0.1:
                    continue

                rgb_px, rgb_py, _ = project_point(rgb_cam_mat, p_opt)
                if not (rgb_py >= rgb.shape[0] or rgb_px >= rgb.shape[1] or rgb_py < 0 or rgb_px < 0): 
                    # Z-Buffer 彩色圖投影
                    if p[2] > self.color_top_down_height[y, x]: 
                        self.color_top_down[y, x] = rgb[rgb_py, rgb_px, :]
                        self.color_top_down_height[y, x] = p[2]
                    cv2.circle(debug_img, (int(rgb_px), int(rgb_py)), 2, (0, 255, 0), -1) # 物理點 (綠)

                px, py, _ = project_point(feat_cam_mat, p_opt)
                if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                    feat = pix_feats[0, :, py, px]
                    # 滑動平均融合 CLIP 特徵
                    self.grid[y, x] = (self.grid[y, x] * self.weight[y, x] + feat) / (self.weight[y, x] + 1) 
                    self.weight[y, x] += 1
                    rgb_px_feat = int(px * scale_x)
                    rgb_py_feat = int(py * scale_y)
                    cv2.circle(debug_img, (rgb_px_feat, rgb_py_feat), 2, (0, 0, 255), -1)       # 語意點 (紅)
                
                # 障礙物判定：離地 10 公分以上標記為 1 (障礙物)
                if p[2] > 0.10: 
                    self.obstacles[y, x] = 1 
            
            if frame_idx % 10 == 0:
                debug_save_dir = os.path.join(self.data_dir, "debug")
                os.makedirs(debug_save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_save_dir, f"debug_{frame_idx:04d}.png"), debug_img)
            pbar.update(1)
        pbar.close()
        
        # 進行邊界裁切與存檔
        self.save_custom_maps()

    def save_custom_maps(self):
        print("[系統] 正在計算地圖邊界並進行裁切...")
        
        # 利用 weight > 0 找出真正有探索到的區域邊界
        explored_coords = np.argwhere(self.weight > 0)
        if len(explored_coords) == 0:
            print("[錯誤] 這是一張空地圖，沒有探索到任何特徵！")
            return
            
        ymin, xmin = explored_coords.min(axis=0)
        ymax, xmax = explored_coords.max(axis=0)
        
        # 賦值給父類別屬性，以便未來與官方函數相容
        self.rmin, self.rmax = ymin, ymax
        self.cmin, self.cmax = xmin, xmax
        
        # 裁切陣列
        grid_cropped = self.grid[ymin:ymax+1, xmin:xmax+1]
        weight_cropped = self.weight[ymin:ymax+1, xmin:xmax+1]
        obstacles_cropped = self.obstacles[ymin:ymax+1, xmin:xmax+1]
        color_cropped = self.color_top_down[ymin:ymax+1, xmin:xmax+1]

        # 儲存
        map_save_dir = os.path.join(self.data_dir, "map")
        os.makedirs(map_save_dir, exist_ok=True)
        
        save_map(os.path.join(map_save_dir, "grid_lseg_1.npy"), grid_cropped)
        save_map(os.path.join(map_save_dir, "weight_lseg_1.npy"), weight_cropped)
        save_map(os.path.join(map_save_dir, "obstacles.npy"), obstacles_cropped)
        cv2.imwrite(os.path.join(map_save_dir, "color_top_down.png"), cv2.cvtColor(color_cropped, cv2.COLOR_RGB2BGR))
        
        print(f"[系統] 裁切範圍 X({xmin}:{xmax}), Y({ymin}:{ymax})")
        print("✅ 終極混血版地圖建立完成並儲存成功！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    
    print(f"[啟動] 載入資料夾: {args.data_dir}")
    
    # 建立 Tracer 專屬的 VLMap 物件並執行建圖
    # 相機高度已調整為適合 Tracer 的 0.69 公尺
    tracer_map = TracerVLMap(data_dir=args.data_dir, camera_height=0.69)
    tracer_map.create_map()