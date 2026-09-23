import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_depth_png_to_npy():
    # 1. 設定路徑
    base_dir = os.path.expanduser('~/vlmaps_ws/dataset')
    depth_dir = Path(base_dir) / "depth"

    # 2. 尋找所有的 .png 深度圖
    png_files = sorted(depth_dir.glob("*.png"))
    
    if not png_files:
        print(f"❌ 找不到任何 .png 檔案於: {depth_dir}")
        return

    print(f"找到 {len(png_files)} 張深度圖，開始轉換並校正單位...")

    flag = True
    # 3. 逐一轉換
    for png_file in tqdm(png_files):
        # 讀取 16-bit 的深度圖 (IMREAD_UNCHANGED 非常重要，否則精度會遺失)
        depth_img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
        
        if depth_img is None:
            print(f"⚠️ 無法讀取: {png_file.name}")
            continue

        # 【關鍵步驟】將單位從 公釐 (mm) 轉換為 公尺 (m)，並轉為 float32 格式
        depth_meters = depth_img.astype(np.float32) / 1000.0
        depth_meters[depth_img == 0] = np.nan  # 強制把無效點變 NaN
        if flag:
            print(f"無效點數量: {len(depth_meters[np.isnan(depth_meters)])}")
            print(f"有效點數量: {len(depth_meters[~np.isnan(depth_meters)])}")
            flag = False

        # 將檔案儲存為同檔名的 .npy
        npy_path = depth_dir / (png_file.stem + ".npy")
        np.save(npy_path, depth_meters)

    print("✅ 深度圖轉換完成！現在 dataset/depth 裡面有 .npy 檔案了。")

if __name__ == "__main__":
    convert_depth_png_to_npy()