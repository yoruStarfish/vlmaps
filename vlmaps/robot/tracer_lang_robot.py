import os
import pickle
import numpy as np
import h5py
from omegaconf import DictConfig, OmegaConf

from vlmaps.map.map import Map

from typing import List, Tuple, Dict, Any

from datetime import datetime
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped, PoseArray
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration
from tf2_geometry_msgs import do_transform_pose
from rclpy.node import Node
from rclpy.logging import get_logger
from rclpy.time import Time
import time
from cv_bridge import CvBridge
import tf_transformations
import os
import cv2
from sensor_msgs.msg import Image
from vlmaps.map.vlmap_tracer import VLMapTracer


class TracerLangRobot:
    """
    This class provides all primitives API that the robot can call during navigation
    """

    def __init__(self, config: DictConfig, navigator=None, localization_node=None, wait_fn=None):
        self.config = config
        self.curr_pos_on_map = None
        self.curr_ang_deg_on_map = None
        
        # 接收從主程式傳入的 ROS 2 實例與等待函式
        self.navigator = navigator
        self.localization_node = localization_node
        self.wait_fn = wait_fn

    def get_vlmaps_data_dirs(self):
        return self.vlmaps_data_save_dirs

    def load_scene_map(self, data_dir: str, map_config: DictConfig):
        # self.map = Map(map_config, data_dir)
        self.map = VLMapTracer(map_config, data_dir)
        self.map.load_map(data_dir)
        self.map.generate_obstacle_map()
        self.map._init_clip() # 記得初始化 CLIP 模型
        categories = [
            "chair",
            "table",
            "wall",
            "cardboard box",
            "sandwich",
            "laptop",
            "paper cup",
            "kettle",
            "bottle",
            "floor",
            "other"
        ]
        self.map.init_categories(categories)

    def empty_recorded_actions(self):
        self.recorded_actions_list = []
        self.recorded_robot_pos = []
        self.goal_tfs = None
        self.all_goal_tfs = None
        self.goal_id = None

    def get_recorded_actions(self):
        return self.recorded_actions_list

    def load_code(self, code_dir: str, task_i: int):
        code_path = os.path.join(code_dir, f"{task_i:06}.txt")
        with open(code_path, "r") as f:
            code = f.readlines()
        code = "".join(code)
        return code


    def _set_nav_curr_pose(self):
        """
        Set self.curr_pos_on_map and self.curr_ang_deg_on_map
        based on the simulator agent ground truth pose

        向 ROS 2 取得當前真實座標，並換算成 VLMaps 網格 (row, col) 與角度
        """
        import rclpy
        import math
        import tf_transformations

        if self.localization_node is None:
            print("未綁定 localization_node，無法更新位置。")
            return

        # 更新 ROS 2 座標
        rclpy.spin_once(self.localization_node, timeout_sec=0.2)
        current_pose = self.localization_node.pose_publisher.get_pose()

        if current_pose is None:
            print("無法取得機器人當前座標。")
            return

        world_x = current_pose.pose.position.x
        world_y = current_pose.pose.position.y

        # 轉換為 VLMaps 網格座標
        grid_size = self.map.gs
        cell_size = self.map.cs
        center_offset = grid_size / 2.0

        col = int((world_x / cell_size) + center_offset)
        row = int((-world_y / cell_size) + center_offset)

        self.curr_pos_on_map = [row, col]

        # 計算當前角度 (Yaw 轉 Degree)
        q = current_pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.curr_ang_deg_on_map = math.degrees(yaw)

    def execute_actions(self, actions_list: List[Any]):
        return NotImplementedError

    def _execute_action(self, action: str):
        return NotImplementedError

    def get_sound_pos(self, name: str):
        return NotImplementedError
        # tfs = self.sound_map.get_pos(name)

    def move_to(self, pos: Tuple[float, float], face_pos: Tuple[float, float] = None):
        """
        接收 VLMaps 網格目標 pos (row, col) 並導航。
        若提供 face_pos (row, col)，抵達時的車頭朝向會正對 face_pos。
        """
        import math
        import tf_transformations

        if pos is None:
            print("找不到目標座標。", flush=True)
            return
            
        row, col = pos
        grid_size = self.map.gs
        cell_size = self.map.cs
        center_offset = grid_size / 2.0

        world_x = (col - center_offset) * cell_size
        world_y = -(row - center_offset) * cell_size

        # 計算終點朝向：若有指定要面向的物體座標，則正對該物體
        if face_pos is not None:
            face_row, face_col = face_pos
            face_x = (face_col - center_offset) * cell_size
            face_y = -(face_row - center_offset) * cell_size
            yaw = math.atan2(face_y - world_y, face_x - world_x)
            q = tf_transformations.quaternion_from_euler(0, 0, yaw)
        else:
            q = [0.0, 0.0, 0.0, 1.0]
            
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(world_x)
        goal_pose.pose.position.y = float(world_y)
        goal_pose.pose.orientation.x = q[0]
        goal_pose.pose.orientation.y = q[1]
        goal_pose.pose.orientation.z = q[2]
        goal_pose.pose.orientation.w = q[3]
        
        print(f"導航至目標座標: ({world_x:.2f}, {world_y:.2f}) 公尺", flush=True)
        self.navigator.goToPose(goal_pose)
        
        if hasattr(self, 'wait_fn') and self.wait_fn is not None:
            self.wait_fn(self.navigator)

    def turn(self, angle_deg: float):
        """
        接收相對旋轉角度，並透過 Nav2 執行原地轉向。
        """
        import math
        import tf_transformations
        from geometry_msgs.msg import PoseStamped
        
        # 1. 取得機器人最新的真實座標與角度
        self._set_nav_curr_pose()
        
        # 2. 計算旋轉後的新絕對角度 (度)
        target_yaw_deg = self.curr_ang_deg_on_map + angle_deg
        target_yaw_rad = math.radians(target_yaw_deg)
        
        # 3. 將角度轉回 ROS 2 要求的四元數 (Quaternion)
        q = tf_transformations.quaternion_from_euler(0, 0, target_yaw_rad)
        
        # 4. 發送原地轉向的 Nav2 指令 (X, Y 維持當前世界座標)
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        
        # 必須將當前的網格座標 (row, col) 轉回真實世界公尺
        grid_size = self.map.gs
        cell_size = self.map.cs
        center_offset = grid_size / 2.0
        row, col = self.curr_pos_on_map
        
        world_x = (col - center_offset) * cell_size
        world_y = -(row - center_offset) * cell_size
        
        goal_pose.pose.position.x = float(world_x)
        goal_pose.pose.position.y = float(world_y)
        
        goal_pose.pose.orientation.x = q[0]
        goal_pose.pose.orientation.y = q[1]
        goal_pose.pose.orientation.z = q[2]
        goal_pose.pose.orientation.w = q[3]
        
        print(f"原地旋轉至目標角度: {target_yaw_deg:.2f} 度", flush=True) 
        self.navigator.goToPose(goal_pose)
        # actions_list = self.nav.turn(angle_deg)
        # self.execute_actions(actions_list)
        # self.actions_list += actions_list

        # 在 move_to 與 turn 的 self.navigator.goToPose(goal_pose) 下一行加入：
        if self.wait_fn is not None:
            self.wait_fn(self.navigator)

    def get_agent_pose_on_map(self) -> Tuple[float, float, float]:
        """
        Return row, col, angle_degree on the full map
        """
        return (
            self.curr_pos_on_map[0],
            self.curr_pos_on_map[1],
            self.curr_ang_deg_on_map,
        )

    def get_pos(self, name: str):
        """
        Return nearest object position on the map
        """
        contours, centers, bbox_list = self.map.get_pos(name)
        if not centers:
            print(f"no objects {name} detected")
            return self.curr_pos_on_map
        ids = self.map.filter_small_objects(bbox_list)
        if ids:
            centers = [centers[x] for x in ids]
            bbox_list = [bbox_list[x] for x in ids]

        nearest_id = self.map.select_nearest_obj(centers, bbox_list, self.curr_pos_on_map)
        center = centers[nearest_id]
        return center

    def get_contour(self, name: str) -> List[List[float]]:
        """
        Return nearest object contour points on the map
        """
        contours, centers, bbox_list = self.map.get_pos(name)
        if not centers:
            print(f"no objects {name} detected")
            assert False
        ids = self.map.filter_small_objects(bbox_list)
        if ids:
            centers = [centers[x] for x in ids]
            bbox_list = [bbox_list[x] for x in ids]
            contours = [contours[x] for x in ids]

        nearest_id = self.map.select_nearest_obj(centers, bbox_list, self.curr_pos_on_map)
        contour = contours[nearest_id]
        return contour

    def with_object_on_left(self, name: str):
        self.face(name)
        self.turn(90)

    def with_object_on_right(self, name: str):
        self.face(name)
        self.turn(-90)

    def move_to_left(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_left_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_to_right(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_right_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_in_between(self, name_a: str, name_b: str):
        self._set_nav_curr_pose()
        pos = self.map.get_pos_in_between(
            self.curr_pos_on_map, self.curr_ang_deg_on_map, name_a, name_b
        )
        
        # 若官方函式因為中點壓在桌面或障礙物上而回傳 None，啟動備援搜尋
        if pos is None:
            print(f"官方中點計算失敗，啟動備援機制計算 {name_a} 與 {name_b} 之間的可通行點...", flush=True)
            pos_a = self.get_pos(name_a)
            pos_b = self.get_pos(name_b)
            
            if pos_a is None or pos_b is None:
                print("無法取得兩個物件的座標。", flush=True)
                return
                
            # 1. 計算兩物件的幾何中點
            mid_row = int(round((pos_a[0] + pos_b[0]) / 2.0))
            mid_col = int(round((pos_a[1] + pos_b[1]) / 2.0))
            
            # 2. 取得 VLMaps 的障礙物地圖 (相容不同版本的屬性名稱)
            obs_map = getattr(self.map, 'obstacles_new', getattr(self.map, 'obstacle_map', None))
            
            if obs_map is not None:
                # 為了避免選到剛好貼在桌邊的像素，將障礙物稍微膨脹 0.35 公尺作為安全邊界
                safe_margin_px = max(1, int(0.35 / self.map.cs))
                kernel = np.ones((safe_margin_px * 2 + 1, safe_margin_px * 2 + 1), np.uint8)
                # obs_map 中 0 代表可通行區域，大於 0 代表障礙物
                occupied = (obs_map > 0).astype(np.uint8)
                dilated_occupied = cv2.dilate(occupied, kernel, iterations=1)
                
                # 找出所有安全的自由空間座標
                free_points = np.argwhere(dilated_occupied == 0)
                if len(free_points) > 0:
                    mid_vec = np.array([mid_row, mid_col])
                    distances = np.linalg.norm(free_points - mid_vec, axis=1)
                    nearest_idx = np.argmin(distances)
                    pos = (int(free_points[nearest_idx][0]), int(free_points[nearest_idx][1]))
                    print(f"原始中點: [{mid_row}, {mid_col}] -> 修正後的安全走道點: {pos}", flush=True)
                else:
                    pos = (mid_row, mid_col)
            else:
                pos = (mid_row, mid_col)

        self.move_to(pos)

    def turn_absolute(self, angle_deg: float):
       """
        將機器人轉向地圖上的絕對角度 (例如 0 度為正東方)
        """
       self._set_nav_curr_pose()
       delta_deg = angle_deg - self.curr_ang_deg_on_map

       # 將角度限制在 -180 到 180 度之間，確保機器人走最短路徑轉向
       delta_deg = np.mod(delta_deg, 360)
       if delta_deg < -180:
            delta_deg += 360
       elif delta_deg > 180:
            delta_deg -= 360
    
       self.turn(delta_deg)

    def face(self, name: str):
        self._set_nav_curr_pose()
        turn_right_angle = self.map.get_delta_angle_to(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.turn(turn_right_angle)

    def move_north(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_north_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_south(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_south_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_west(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_west_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_east(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_east_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_to_object(self, name: str, safe_distance_m: float = 0.55):
        self._set_nav_curr_pose()
        
        # 1. 取得物體邊緣距離機器人最近的點
        edge_pos = self.map.get_nearest_pos(self.curr_pos_on_map, name)

        if edge_pos == self.curr_pos_on_map:
            print(f"找不到物件: {name}，或是已在該位置。", flush=True)
            return
            
        # 2. 將安全公尺數轉換為網格像素數
        safe_pixels = safe_distance_m / self.map.cs
        
        # 3. 計算從「當前位置」指向「物體邊緣」的向量
        curr_vec = np.array(self.curr_pos_on_map, dtype=float)
        edge_vec = np.array(edge_pos, dtype=float)
        
        direction = edge_vec - curr_vec
        distance = np.linalg.norm(direction)
        
        # 若已經在安全距離內，直接原地轉向面對該物體即可
        if distance <= safe_pixels:
            print("已經在安全觀測距離內，直接轉向目標物。", flush=True)
            self.face(name)
            return
            
        # 4. 沿著連線往後退 safe_pixels 距離
        direction_normalized = direction / distance
        safe_pos = edge_vec - (direction_normalized * safe_pixels)
        
        final_row = int(round(safe_pos[0]))
        final_col = int(round(safe_pos[1]))
        
        print(f"物體邊緣點: {edge_pos}, 往後退 {safe_distance_m}m 的安全點: [{final_row}, {final_col}]", flush=True)
        
        # 5. 發送給 Nav2，並傳入 edge_pos 讓機器人抵達時正對目標物
        self.move_to((final_row, final_col), face_pos=edge_pos)

    def move_forward(self, meters: float):
        self._set_nav_curr_pose()
        pos = self.map.get_forward_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, meters)
        self.move_to(pos)
