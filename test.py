# import h5py
# import numpy as np

# file_path = 'dataset/vlmap/vlmaps.h5df'

# with h5py.File(file_path, 'r') as f:
#     for key in f.keys():
#         dataset = f[key]
#         print(f"Key: {key}")
#         print(f"  Shape: {dataset.shape}")
#         print(f"  Dtype: {dataset.dtype}")
        
#         # 如果是小型的 dataset，可以印出部分數值來觀察
#         if dataset.shape[0] < 10:
#             print(f"  Sample data: {dataset[:]}")
#         else:
#             print(f"  Sample data: {dataset[:5]} ... (truncated)")
#         print("-" * 20)

# import h5py
# import matplotlib.pyplot as plt

# file_path = '/home/robotic/vlmaps_ws/dataset/vlmap/vlmaps.h5df'

# try:
#     with h5py.File(file_path, 'r') as f:
#         # 讀取必要數據
#         pos = f['grid_pos'][:]      # 形狀為 (N, 3)
#         rgb = f['grid_rgb'][:]      # 形狀為 (N, 3)
        
#         # 提取 X 和 Y 座標
#         x = pos[:, 0]
#         y = pos[:, 1]
        
#         # 將 RGB 轉為 matplotlib 可用的 0-1 格式
#         colors = rgb / 255.0
        
#         # 開始繪圖
#         plt.figure(figsize=(10, 8))
#         plt.scatter(x, y, c=colors, s=5, marker='.')
        
#         plt.title("VLMap Visualization (Top-Down View)")
#         plt.xlabel("Grid X")
#         plt.ylabel("Grid Y")
#         plt.gca().invert_yaxis() # 有時候座標系原點在左上，這行可以根據實際情況取消註解
#         plt.grid(True, linestyle='--', alpha=0.3)
#         plt.axis('equal') # 確保長寬比一致
        
#         plt.show()

# except Exception as e:
#     print(f"視覺化過程中發生錯誤: {e}")

# import h5py
# import matplotlib.pyplot as plt
# import numpy as np

# file_path = '/home/robotic/vlmaps_ws/dataset/vlmap/vlmaps.h5df'

# with h5py.File(file_path, 'r') as f:
#     # 讀取佔用數據
#     # 假設我們只取第三維度中的第一層作為基礎地圖 (你可以嘗試調整 index 0-12)
#     occ = f['occupied_ids'][:, :, 0] 
#     occupied_count = np.sum(occ != -1)
#     print(f"總網格數: {occ.size}")
#     print(f"已佔用網格數: {occupied_count}")
#     print(f"佔用比例: {occupied_count / occ.size * 100:.2f}%")

#     pos = f['grid_pos'][:]
#     print(f"X 座標範圍: {pos[:, 0].min()} ~ {pos[:, 0].max()}")
#     print(f"Y 座標範圍: {pos[:, 1].min()} ~ {pos[:, 1].max()}")
    
#     # -1 代表未探索區域，我們將其過濾掉，只看有數據的區域
#     # 將 -1 轉為 0，其他標籤設為 1，做一個二值化地圖
#     map_view = np.where(occ != -1, 1, 0)
    
#     plt.figure(figsize=(10, 10))
#     plt.imshow(map_view, cmap='gray', origin='lower')
#     plt.title("Occupancy Grid Map (Full View)")
#     plt.colorbar(label='Occupied (1) / Free/Unknown (0)')
#     plt.show()

import matplotlib.pyplot as plt
import numpy as np
load_txt_path = '/home/robotic/vlmaps_ws/dataset/poses.txt'
# 讀取 poses.txt
poses = np.loadtxt(load_txt_path)

# 提取所有的 X, Y 座標
x_coords = poses[:, 0] # 假設第 2 個欄位是 X
y_coords = poses[:, 1] # 假設第 3 個欄位是 Y

plt.figure(figsize=(10, 8))
plt.plot(x_coords, y_coords, marker='o', markersize=2, linestyle='-', linewidth=0.5)
plt.title("RTAB-Map Exported Trajectory")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True)
plt.axis('equal') # 重要：保持 1:1 比例，否則會變形
plt.savefig("trajectory_visualization.png")
plt.show()