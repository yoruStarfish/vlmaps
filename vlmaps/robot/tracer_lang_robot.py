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
from navigatethroughposes.navigatethroughposes.navi_vlmaps import LocalizationNode


class TracerLangRobot:
    """
    This class provides all primitives API that the robot can call during navigation
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.curr_pos_on_map = None
        self.curr_ang_deg_on_map = None
        # 綁定 ROS 2 的導航與定位元件

        self.navigator = BasicNavigator()
        self.localization_node = LocalizationNode()

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

    def move_to(self, pos: Tuple[float, float]):
        """
        Move the robot to the position on the map
        based on accurate localization in the environment
        """
        """
        接收 VLMaps 網格目標 (row, col)，轉換為公尺並透過 Nav2 導航
        """
        if pos is None:
            print("找不到目標座標。")
            return
            
        row, col = pos

        # 將網格轉換回真實世界公尺
        grid_size = self.map.gs
        cell_size = self.map.cs
        center_offset = grid_size / 2.0

        world_x = (col - center_offset) * cell_size
        world_y = -(row - center_offset) * cell_size
            
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = float(world_x)
        goal_pose.pose.position.y = float(world_y)
        goal_pose.pose.orientation.w = 1.0
        self.get_logger().info(f"導航至目標座標: ({world_x:.2f}, {world_y:.2f}) 公尺")
        self.navigator.goToPose(goal_pose)

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
        
        self.get_logger().info(f"原地旋轉至目標角度: {target_yaw_deg:.2f} 度") 
        self.navigator.goToPose(goal_pose)
        # actions_list = self.nav.turn(angle_deg)
        # self.execute_actions(actions_list)
        # self.actions_list += actions_list

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
        pos = self.map.get_pos_in_between(self.curr_pos_on_map, self.curr_ang_deg_on_map, name_a, name_b)
        self.move_to(pos)

    def turn_absolute(self, angle_deg: float):
        self._set_nav_curr_pose()
        delta_deg = angle_deg - self.curr_ang_deg_on_map
        actions_list = self.turn(delta_deg)
        self.recorded_actions_list.extend(actions_list)

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

    def move_to_object(self, name: str, safe_distance_m: float = 0.5):
        self._set_nav_curr_pose()
        # 1. 呼叫官方 API，取得「剛好貼在物體邊緣」的點
        edge_pos = self.map.get_nearest_pos(self.curr_pos_on_map, name)

        if edge_pos == self.curr_pos_on_map:
            print(f"找不到物件: {name}，或是已在該位置。")
            return
            
        # 2. 將安全公尺數轉換為地圖網格的像素數量
        safe_pixels = safe_distance_m / self.map.cs
        
        # 3. 計算從「機器人當前位置」指向「物體邊緣」的向量
        curr_vec = np.array(self.curr_pos_on_map)
        edge_vec = np.array(edge_pos)
        
        direction = edge_vec - curr_vec
        distance = np.linalg.norm(direction)
        
        # 如果機器人距離物體已經比安全距離還近，就不移動
        if distance <= safe_pixels:
            print("已經在安全觀測距離內，無需移動。")
            return
            
        # 4. 將向量正規化 (變成長度為 1 的單位向量)
        direction_normalized = direction / distance
        
        # 5. 計算退後 safe_pixels 距離後的新座標
        safe_pos = edge_vec - (direction_normalized * safe_pixels)
        
        # 轉回整數網格座標
        final_row = int(round(safe_pos[0]))
        final_col = int(round(safe_pos[1]))
        
        print(f"物體邊緣點: {edge_pos}, 往後退 {safe_distance_m}m 的安全點: [{final_row}, {final_col}]")
        
        # 6. 發送給 Nav2
        self.move_to((final_row, final_col))

    def move_forward(self, meters: float):
        self._set_nav_curr_pose()
        pos = self.map.get_forward_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, meters)
        self.move_to(pos)
