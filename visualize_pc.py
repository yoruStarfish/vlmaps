import open3d as o3d
import os

def view_ply(file_path):
    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 {file_path}")
        return

    print("正在載入點雲...")
    # 讀取 ply 檔案
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 檢查是否成功讀取到點
    if not pcd.has_points():
        print("錯誤：點雲檔案中沒有任何點資料！")
        return
        
    print(f"成功載入點雲！總共包含 {len(pcd.points)} 個點。")
    print("正在開啟視覺化視窗... (提示：在視窗中按 'h' 可以查看操作說明)")
    
    # 開啟 3D 視覺化視窗
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Open3D - RTAB-Map Point Cloud Viewer",
        width=1024,
        height=768,
        left=50,
        top=50,
        mesh_show_back_face=False
    )

if __name__ == "__main__":
    # 💡 請將這裡替換成您實際的 .ply 檔案路徑
    ply_file_path = "/home/robotic/.ros/rtabmap_cloud.ply" 
    
    # view_ply(ply_file_path)

    # 讀取你的 pcd 檔案
    pcd = o3d.io.read_point_cloud("debug_full_global_colored_map.pcd")
    # pcd = o3d.io.read_point_cloud("debug_0_ultimate_colored.pcd")
    # 視覺化
    o3d.visualization.draw_geometries([pcd])