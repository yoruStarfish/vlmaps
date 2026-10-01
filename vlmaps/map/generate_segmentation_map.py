import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from omegaconf import OmegaConf

base_dir = os.path.expanduser('~/vlmaps_ws')
vlmaps_src_dir = os.path.join(base_dir, 'vlmapsSrc')



# 將兩個層級的路徑都加入，確保不會再發生 ModuleNotFoundError
if vlmaps_src_dir not in sys.path:
    sys.path.insert(0, vlmaps_src_dir)

from vlmaps.map.vlmap_tracer import VLMapTracer
from vlmaps.utils.visualize_utils import pool_3d_label_to_2d

# 1. 定義你要的所有類別
categories = [
    "floor",         # 建議加入地板，吸收掉地面的雜訊
    "wall",
    "table",
    "laptop",
    "sandwich",
    "rubber mallet",
    "mallet",
    "spatula",
    "paper cup",
    "plate",
    "cardboard box",
    "bottle",
]

# 2. 為每個類別指定專屬的 RGB 顏色 (0~255)
# 這裡參考了論文的配色風格
color_dict = {
    "floor": [0, 128, 0],          # 綠色
    "wall": [128, 0, 0],           # 深紅色
    "table": [128, 0, 128],        # 紫色
    "laptop": [255, 165, 0],       # 橘色
    "sandwich": [255, 0, 0],       # 紅色
    "rubber mallet": [128, 128, 0],       # 橄欖色
    "mallet": [0, 255, 0],  # 綠色
    "spatula": [0, 255, 255],      # 青色
    "paper cup": [255, 255, 0],          # 黃色
    "plate": [255, 0, 255],        # 洋紅色
    "cardboard box": [205, 133, 63], # 棕色/紙箱色
    "bottle": [0, 0, 255],          # 藍色
}
base_dir = os.path.expanduser('~/vlmaps_ws')
map_config = OmegaConf.load("vlmapsSrc/config/map_config/vlmaps_tracer.yaml")
data_dir = os.path.join(base_dir, "dataset")
vlmap_tracer = VLMapTracer(map_config, data_dir)
vlmap_tracer.load_map(data_dir)
vlmap_tracer.generate_obstacle_map()
vlmap_tracer.customize_obstacle_map(
    potential_obstacle_names=map_config.potential_obstacle_names,
    obstacle_names=map_config.obstacle_names
)
vlmap_tracer._init_clip() # 記得初始化 CLIP 模型

# 初始化 CLIP 類別
vlmap_tracer.init_categories(categories=categories)

# 3. 準備畫布
# 取得真實障礙物的範圍
obs_map_cropped = vlmap_tracer.obstacles_new_cropped
H, W = obs_map_cropped.shape

# 建立一張淺灰色的畫布 (代表未探索的區域)
canvas = np.ones((H, W, 3), dtype=np.uint8) * 220 

# 🚨 終極修正：正確定義「障礙物」與「空地」
# 在影像矩陣中，0 代表黑色(障礙物)，大於 0 代表白色(空地)
is_obstacle = (obs_map_cropped == 0)
is_free_space = (obs_map_cropped > 0)

# 4. 依序將每個類別的遮罩畫上畫布
for cat in categories:
    # 取得 1D 預測結果並轉為 2D 裁切
    mask_1d = vlmap_tracer.index_map(cat)
    mask_2d = pool_3d_label_to_2d(mask_1d, vlmap_tracer.grid_pos, vlmap_tracer.gs)
    mask_2d_cropped = mask_2d[vlmap_tracer.rmin : vlmap_tracer.rmax + 1, 
                              vlmap_tracer.cmin : vlmap_tracer.cmax + 1]
    
    # 🌟 嚴格物理約束：什麼東西該畫在哪裡？
    if cat == "floor":
        # 地板：必須被 AI 判定為 floor，且該位置在幾何地圖上必須是「空地」
        valid_pixels = (mask_2d_cropped > 0) & is_free_space
    else:
        # 其他所有物品 (牆壁、桌椅等)：必須被 AI 判定為該物品，且該位置必須是「障礙物」
        valid_pixels = (mask_2d_cropped > 0) & is_obstacle
    
    # 將對應的像素塗上專屬顏色
    color = color_dict.get(cat, [0, 0, 0])
    canvas[valid_pixels] = color

# 5. 使用 Matplotlib 繪製並加上 Legend (圖例)
fig, ax = plt.subplots(figsize=(10, 10), dpi=200) # dpi=200 讓圖片變高畫質
ax.imshow(canvas)
ax.set_title("VLMaps Semantic Segmentation", fontsize=16)
ax.axis('off') # 隱藏 X, Y 軸的刻度

# 建立圖例 (Legend)
legend_patches = []
for cat in categories:
    # 將 RGB (0~255) 轉為 Matplotlib 需要的 (0~1) 範圍
    color_normalized = [c / 255.0 for c in color_dict[cat]]
    patch = mpatches.Patch(color=color_normalized, label=cat)
    legend_patches.append(patch)

# 將圖例放在圖片外部的右側
ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)

# 6. 儲存圖片 (bbox_inches='tight' 確保圖例不會被切掉)
output_filename = "debug_final_paper_ready_map.png"
plt.savefig(output_filename, bbox_inches='tight', facecolor='white')
print(f"✅ 學術級多類別分割圖已儲存: {output_filename}")