import os
import shutil
from pathlib import Path

def prepare_keyframe_data():
    # 路徑設定
    src_rgb = Path("data/rgb")
    src_depth = Path("data/depth")
    poses_file = Path("data/poses_rtab.txt")
    
    out_dir = Path("dataset")
    out_rgb = out_dir / "rgb"
    out_depth = out_dir / "depth"
    
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)
    
    new_poses = []
    valid_ids = []

    print("正在讀取並過濾關鍵幀 Pose...")
    with open(poses_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            
            parts = line.split()
            # 格式: timestamp, x, y, z, qx, qy, qz, qw, id
            x, y, z = parts[1:4]
            qx, qy, qz, qw = parts[4:8]
            node_id = parts[8]
            
            # 將 qx, qy, qz, qw 存成一行，後續寫入新檔案
            new_poses.append(f"{x} {y} {z} {qx} {qy} {qz} {qw}\n")
            valid_ids.append(node_id)
            
            # 複製對應的圖
            shutil.copy(src_rgb / f"{node_id}.png", out_rgb / f"{node_id}.png")
            shutil.copy(src_depth / f"{node_id}.png", out_depth / f"{node_id}.png")

    # 寫入新的 poses_quat.txt
    with open(out_dir / "poses.txt", 'w') as f:
        f.writelines(new_poses)
        
    print(f"✅ 完成！已提取 {len(valid_ids)} 個關鍵幀至 data_keyframe/")
    print(f"✅ Pose 檔已儲存為 data_keyframe/poses_quat.txt")

if __name__ == "__main__":
    prepare_keyframe_data()