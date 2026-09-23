import os
import sys
import numpy as np
import cv2
import torch
import clip
base_dir = os.path.expanduser('~/vlmaps_ws')
# 加上這行，拯救所有的 import！
vlmaps_core_dir = os.path.join(base_dir, 'vlmapsSrc', 'vlmaps')
sys.path.append(vlmaps_core_dir)

# # 動態將當前腳本的上一層目錄加入系統路徑，預防找不到 utils 模組
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

from utils.clip_utils import get_text_feats

def generate_nav_maps(map_save_dir, target_object="chair"):
    # --- 1. 定義你實驗室的環境類別 ---
    # 請確保 "floor" (地板) 也在裡面，這樣才能區分哪裡可以走
    # 這裡的順序很重要，index 會對應到這些單字
    categories = ["wall", "chair", "table", "floor", "other"]
    
    if target_object not in categories:
        print(f"[錯誤] 目標物 '{target_object}' 不在類別清單內！")
        return

    # --- 2. 載入我們建好的 VLMaps 檔案 ---
    print(f"[系統] 正在載入地圖資料: {map_save_dir}")
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
        print("[錯誤] 這是一張空地圖，沒有探索到任何特徵！")
        return
        
    ymin, xmin = explored_coords.min(axis=0)
    ymax, xmax = explored_coords.max(axis=0)
    
    # 稍微往外擴張一點點邊界 (padding)
    padding = 10
    ymin, ymax = max(0, ymin - padding), min(grid.shape[0] - 1, ymax + padding)
    xmin, xmax = max(0, xmin - padding), min(grid.shape[1] - 1, xmax + padding)

    print(f"[系統] 自動裁切範圍: X({xmin}:{xmax}), Y({ymin}:{ymax})")
    
    grid_cropped = grid[ymin:ymax+1, xmin:xmax+1]
    obstacles_cropped = obstacles[ymin:ymax+1, xmin:xmax+1]
    weight_cropped = weight[ymin:ymax+1, xmin:xmax+1]
    
    # 【關鍵修正】：分離未探索區域與障礙物的邏輯
    unexplored_mask = weight_cropped == 0       # 從未有任何點雲落入的格子
    obstacle_mask = obstacles_cropped == 1      # 確定有實體障礙物的格子
    no_go_mask = unexplored_mask | obstacle_mask # 兩者合併為不可通行區域

    # --- 4. 準備 CLIP 模型與文字特徵 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[系統] 載入 CLIP 模型中 (使用 {device})...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    text_feats = get_text_feats(categories, clip_model, clip_feat_dim=512, batch_size=64)

    # --- 5. 矩陣內積：計算每個網格最像哪個單字 ---
    print("[系統] 正在進行地圖語意檢索...")
    map_feats = grid_cropped.reshape((-1, grid_cropped.shape[-1]))
    scores_list = map_feats @ text_feats.T 
    
    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape((ymax - ymin + 1, xmax - xmin + 1))

    # --- 6. 產出給 Nav2 導航用的兩張地圖 ---
    
    nav_obstacle_map = np.ones_like(obstacles_cropped, dtype=np.uint8) * 255
    nav_obstacle_map[no_go_mask] = 0 # 將不可通行區域塗黑
    cv2.imwrite(os.path.join(map_save_dir, "nav_obstacle_map.png"), nav_obstacle_map)

    target_idx = categories.index(target_object)
    target_mask = (predicts == target_idx).astype(np.uint8) * 255
    target_mask[no_go_mask] = 0 # 將不可通行區域的目標塗黑
    cv2.imwrite(os.path.join(map_save_dir, f"target_mask_{target_object}.png"), target_mask)

    # --- 7. 產出給 Nav2 導航用的.yaml ---

    # 假設你在 VLMaps 建圖時使用的參數是預設的
    cs = 0.05
    gs = 1000
    
    # 計算地圖左下角在真實世界的 (X, Y) 座標
    origin_x = (-gs * cs / 2.0) + (xmin * cs)
    origin_y = (-gs * cs / 2.0) + (ymin * cs)

    # 注意：這裡的字串內容必須嚴格靠左對齊，否則 YAML 解析會失敗
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
        
    print(f"[成功] 生成 Nav2 專用設定檔: {yaml_path}")
    print(f"[成功] 導航地圖產出完成！已存入 {map_save_dir}")
    print(f" -> 障礙物地圖: nav_obstacle_map.png")
    print(f" -> '{target_object}' 的目標地圖: target_mask_{target_object}.png")

if __name__ == "__main__":
    # 使用相對路徑：假設此程式與 dataset 資料夾存在於特定的相對位置
    # 這裡的設計會自動抓取本程式所在的資料夾，並根據你專案的架構去尋找 map 資料夾
    # 假設此腳本放在 vlmapsSrc 中，而資料在 dataset/map 中，我們退回上一層再進入 dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    map_folder = os.path.join(base_dir, "../dataset/map")
    
    # 如果你的 map 檔案是直接放在這個腳本旁邊的 map 資料夾，請改成：
    # map_folder = os.path.join(base_dir, "map")
    
    # 將絕對路徑標準化，避免出現類似 'vlmapsSrc/../dataset/map' 的冗贅字串
    map_folder = os.path.abspath(map_folder)
    
    generate_nav_maps(map_folder, target_object="chair")