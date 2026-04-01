import os
import numpy as np
import cv2
import torch
import clip
from utils.clip_utils import get_text_feats

def generate_nav_maps(map_save_dir, target_object="chair"):
    # --- 1. 定義你實驗室的環境類別 ---
    # 請確保 "floor" (地板) 也在裡面，這樣才能區分哪裡可以走
    # 這裡的順序很重要，index 會對應到這些單字
    categories = ["door", "chair", "desk", "floor", "wall", "other"]
    
    if target_object not in categories:
        print(f"❌ 錯誤：目標物 '{target_object}' 不在類別清單內！")
        return

    # --- 2. 載入我們建好的 VLMaps 檔案 ---
    print(f"📂 正在載入地圖資料: {map_save_dir}")
    grid_path = os.path.join(map_save_dir, "grid_lseg_1.npy")
    obstacles_path = os.path.join(map_save_dir, "obstacles.npy")
    weight_path = os.path.join(map_save_dir, "weight_lseg_1.npy")

    grid = np.load(grid_path)
    obstacles = np.load(obstacles_path)
    weight = np.load(weight_path)

    # --- 3. 自動裁切地圖 (找出機器人實際有探索到的範圍) ---
    # 利用 weight > 0 來找出有資料的網格
    explored_coords = np.argwhere(weight > 0)
    if len(explored_coords) == 0:
        print("❌ 錯誤：這是一張空地圖，沒有探索到任何特徵！")
        return
        
    ymin, xmin = explored_coords.min(axis=0)
    ymax, xmax = explored_coords.max(axis=0)
    
    # 稍微往外擴張一點點邊界 (padding)
    padding = 10
    ymin, ymax = max(0, ymin - padding), min(grid.shape[0] - 1, ymax + padding)
    xmin, xmax = max(0, xmin - padding), min(grid.shape[1] - 1, xmax + padding)

    print(f"✂️ 自動裁切範圍: X({xmin}:{xmax}), Y({ymin}:{ymax})")
    
    grid_cropped = grid[ymin:ymax+1, xmin:xmax+1]
    obstacles_cropped = obstacles[ymin:ymax+1, xmin:xmax+1]
    
    # 障礙物遮罩 (obstacles 陣列中: 0是障礙物, 1是自由空間)
    # 我們把沒有探索到的區域也視為障礙物
    no_map_mask = obstacles_cropped == 0 

    # --- 4. 準備 CLIP 模型與文字特徵 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 載入 CLIP 模型中 (使用 {device})...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # lang_tokens = clip.tokenize(categories).to(device)
    # with torch.no_grad():
    #     text_feats = clip_model.encode_text(lang_tokens)
    #     text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    # text_feats = text_feats.cpu().numpy()
    text_feats = get_text_feats(categories, clip_model, clip_feat_dim=512, batch_size=64)

    # --- 5. 矩陣內積：計算每個網格最像哪個單字 ---
    print("🔍 正在進行地圖語意檢索...")
    map_feats = grid_cropped.reshape((-1, grid_cropped.shape[-1]))
    scores_list = map_feats @ text_feats.T # S = Q @ E^T，這裡 Q 是地圖特徵，E 是文字特徵
    
    # 找出最高分的 index
    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape((ymax - ymin + 1, xmax - xmin + 1))

    # --- 6. 產出給 Nav2 導航用的兩張地圖 ---
    
    # A. 產出導航障礙物地圖 (Occupancy Grid)
    # 為了給 ROS 2 Nav2 使用，我們把: 障礙物=0(黑), 可通行=255(白)
    nav_obstacle_map = np.ones_like(obstacles_cropped, dtype=np.uint8) * 255
    nav_obstacle_map[no_map_mask] = 0
    cv2.imwrite(os.path.join(map_save_dir, "nav_obstacle_map.png"), nav_obstacle_map)

    # B. 產出目標物遮罩 (Target Mask)
    target_idx = categories.index(target_object)
    target_mask = (predicts == target_idx).astype(np.uint8) * 255
    
    # 把屬於障礙物的地方從目標遮罩中剔除 (避免目標點出現在牆壁或未探索區域裡面)
    target_mask[no_map_mask] = 0
    cv2.imwrite(os.path.join(map_save_dir, f"target_mask_{target_object}.png"), target_mask)

    # --- 7. 產出給 Nav2 導航用的.yaml ---

    # 假設你在 VLMaps 建圖時使用的參數是預設的
    cs = 0.05
    gs = 1000
    
    # 計算地圖左下角在真實世界的 (X, Y) 座標
    # 注意：因為我們前面有做「自動裁切 (Cropping)」，所以原點也要跟著偏移！
    # 原本的左下角是 -25.0，現在我們切掉了 xmin 和 ymin 個網格
    origin_x = (-gs * cs / 2.0) + (xmin * cs)
    origin_y = (-gs * cs / 2.0) + (ymin * cs)

    yaml_content = f"""image: nav_obstacle_map.png
        resolution: {cs}
        origin: [{origin_x}, {origin_y}, 0.000000]
        negate: 0
        occupied_thresh: 0.65
        free_thresh: 0.196
        """
    
    yaml_path = os.path.join(map_save_dir, "nav_obstacle_map.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"📄 成功生成 Nav2 專用設定檔: {yaml_path}")

    print(f"✅ 導航地圖產出完成！已存入 {map_save_dir}")
    print(f"➡️ 障礙物地圖: nav_obstacle_map.png")
    print(f"➡️ '{target_object}' 的目標地圖: target_mask_{target_object}.png")


if __name__ == "__main__":
    # 假設這是你剛才建好的地圖資料夾
    map_folder = "/home/robotic/vlmaps/map"
    
    # 我們試著找出 "chair"
    generate_nav_maps(map_folder, target_object="chair")