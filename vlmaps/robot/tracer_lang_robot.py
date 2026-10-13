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

    def __init__(self, config: DictConfig, navigator=None, localization_node=None, wait_fn=None,
                 robot_radius_m=0.25, robot_clearance_m=0.05, robot_footprint_m=None):
        self.config = config
        self.curr_pos_on_map = None
        self.curr_ang_deg_on_map = None
        
        # 接收從主程式傳入的 ROS 2 實例與等待函式
        self.navigator = navigator
        self.localization_node = localization_node
        self.wait_fn = wait_fn
        self.current_world_pose = None
        # Match these to the real robot and BOTH Nav2 costmaps.
        self.between_robot_radius_m = robot_radius_m
        self.between_clearance_m = robot_clearance_m
        self.robot_footprint_m = robot_footprint_m

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
        based on the ROS localization estimate (not ground truth)

        向 ROS 2 取得當前真實座標，並換算成 VLMaps 網格 (row, col) 與角度
        """
        import rclpy
        import math
        import tf_transformations

        if self.localization_node is None:
            raise RuntimeError("未綁定 localization_node，無法更新位置。")

        # 更新 ROS 2 座標
        rclpy.spin_once(self.localization_node, timeout_sec=0.2)
        current_pose = self.localization_node.pose_publisher.get_pose()

        if current_pose is None:
            raise RuntimeError("無法取得機器人當前座標；中止指令。")

        if current_pose.header.frame_id != "map":
            raise RuntimeError("定位 PoseStamped 必須使用 map 座標系。")
        self.current_world_pose = current_pose
        world_x = current_pose.pose.position.x
        world_y = current_pose.pose.position.y

        # 轉換為 VLMaps 網格座標
        grid_size = self.map.gs
        cell_size = self.map.cs
        center_offset = grid_size / 2.0

        col = int(round((world_x / cell_size) + center_offset))
        row = int(round((-world_y / cell_size) + center_offset))

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

    def _grid_to_world(self, pos):
        row, col = pos
        return ((col - self.map.gs / 2.0) * self.map.cs,
                -(row - self.map.gs / 2.0) * self.map.cs)

    def _make_goal(self, pos, face_pos=None):
        import math
        x, y = self._grid_to_world(pos)
        yaw = 0.0
        if face_pos is not None:
            fx, fy = self._grid_to_world(face_pos)
            yaw = math.atan2(fy - y, fx - x)
        q = tf_transformations.quaternion_from_euler(0, 0, yaw)
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.navigator.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y = float(x), float(y)
        (goal.pose.orientation.x, goal.pose.orientation.y,
         goal.pose.orientation.z, goal.pose.orientation.w) = q
        return goal

    def _navigate_and_wait(self, goal, timeout_s=120.0):
        """Raise on failure so a sequential instruction stops at this step.

        wait_fn must return TaskResult.SUCCEEDED, or leave a completed Nav2
        task for getResult(). Early cancellation is treated as failure.
        """
        import rclpy
        accepted = self.navigator.goToPose(goal)
        if accepted is False:
            raise RuntimeError("Nav2 拒絕導航目標；中止指令。")
        if self.wait_fn is not None:
            result = self.wait_fn(self.navigator)
            if result is None and self.navigator.isTaskComplete():
                result = self.navigator.getResult()
        else:
            deadline = time.monotonic() + timeout_s
            while rclpy.ok() and not self.navigator.isTaskComplete():
                if time.monotonic() >= deadline:
                    self.navigator.cancelTask()
                    raise RuntimeError("導航逾時；中止後續動作。")
                time.sleep(0.05)
            result = self.navigator.getResult() if rclpy.ok() else None
        if result != TaskResult.SUCCEEDED:
            self.navigator.cancelTask()
            raise RuntimeError(f"導航未成功（{result}）；中止後續動作。")
        return result

    def move_to(self, pos: Tuple[float, float], face_pos=None):
        if pos is None:
            raise RuntimeError("找不到目標座標；中止指令。")
        goal = self._make_goal(pos, face_pos)
        print(f"導航至 ({goal.pose.position.x:.2f}, "
              f"{goal.pose.position.y:.2f}) m", flush=True)
        return self._navigate_and_wait(goal)

    def _refresh_rotation_pose(self, max_tf_age_s=2.0):
        """Read a fresh composed TF instead of the localization timer cache."""
        import math
        import rclpy
        if self.localization_node is None:
            raise RuntimeError("未綁定 localization_node，無法更新旋轉定位。")
        rclpy.spin_once(self.localization_node, timeout_sec=0.01)
        trans = self.localization_node.tf_buffer.lookup_transform(
            'map', 'base_link', Time(), timeout=Duration(seconds=0.1))
        age = (self.localization_node.get_clock().now() -
               Time.from_msg(trans.header.stamp)).nanoseconds / 1e9
        if not math.isfinite(age) or age < -0.1 or age > max_tf_age_s:
            raise RuntimeError(f"旋轉 TF 過期或時間異常：{age:.2f}s")
        t, q = trans.transform.translation, trans.transform.rotation
        values = [t.x, t.y, q.x, q.y, q.z, q.w]
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError("旋轉定位含無效座標。")
        if sum(v*v for v in values[2:]) < 1e-12:
            raise RuntimeError("旋轉定位四元數無效。")
        _, _, yaw = tf_transformations.euler_from_quaternion(values[2:])
        pose = PoseStamped()
        pose.header = trans.header
        pose.pose.position.x, pose.pose.position.y = t.x, t.y
        pose.pose.orientation = q
        self.current_world_pose = pose
        self.curr_pos_on_map = [int(round(self.map.gs/2-t.y/self.map.cs)),
                                int(round(self.map.gs/2+t.x/self.map.cs))]
        self.curr_ang_deg_on_map = math.degrees(yaw)
        return yaw

    def _rotate_to_yaw(self, target_yaw, timeout_s=60.0, max_wz=0.8,
                       yaw_tolerance_deg=3.0):
        """Closed-loop cmd_vel rotation, upstream of collision_monitor.

        Uses the same topic as path.py:rotate_to_yaw. Nav2 navigation must
        already be complete; no navigation or Spin action is sent here.
        """
        import math
        import rclpy
        from geometry_msgs.msg import Twist
        if (not all(math.isfinite(v) for v in
                    (target_yaw, timeout_s, max_wz, yaw_tolerance_deg))
                or min(timeout_s, max_wz, yaw_tolerance_deg) <= 0):
            raise ValueError("旋轉參數無效。")
        if self.navigator is None or not self.navigator.isTaskComplete():
            raise RuntimeError("請先結束目前導航，再執行 cmd_vel 旋轉。")
        publisher = self.navigator.create_publisher(Twist, '/cmd_vel_filtered', 10)
        tolerance = math.radians(yaw_tolerance_deg)
        deadline = time.monotonic() + timeout_s
        settled = 0
        next_log = 0.0
        try:
            # Allow discovery and the upstream smoother/filter to finish
            # publishing their final stop before taking control of this topic.
            ready_at = time.monotonic() + 1.0
            while rclpy.ok() and time.monotonic() < ready_at:
                publisher.publish(Twist())
                rclpy.spin_once(self.navigator, timeout_sec=0.01)
                time.sleep(0.04)
            while rclpy.ok():
                cycle_start = time.monotonic()
                if cycle_start >= deadline:
                    raise RuntimeError("cmd_vel 旋轉逾時；中止後續動作。")
                # Any TF failure exits through finally and immediately stops.
                yaw = self._refresh_rotation_pose()
                error = math.atan2(math.sin(target_yaw-yaw), math.cos(target_yaw-yaw))
                command = Twist()
                if abs(error) <= tolerance:
                    settled += 1
                else:
                    settled = 0
                    command.angular.z = max(-max_wz, min(max_wz, error))
                publisher.publish(command)
                if settled >= 3:
                    return TaskResult.SUCCEEDED
                if cycle_start >= next_log:
                    print(f"cmd_vel 旋轉: 目前={math.degrees(yaw):.1f}°，"
                          f"誤差={math.degrees(error):.1f}°，"
                          f"wz={command.angular.z:.3f}", flush=True)
                    next_log = cycle_start + 0.5
                time.sleep(max(0.0, 0.05-(time.monotonic()-cycle_start)))
            raise RuntimeError("ROS 已停止；旋轉中止。")
        finally:
            # Stop on success, timeout, TF exception and keyboard interruption.
            try:
                if rclpy.ok():
                    for _ in range(3):
                        publisher.publish(Twist())
                        time.sleep(0.02)
            finally:
                self.navigator.destroy_publisher(publisher)

    def turn(self, angle_deg: float):
        """Turn by the shortest equivalent relative angle using cmd_vel."""
        import math
        if not math.isfinite(angle_deg):
            raise ValueError("旋轉角度必須是有限數值。")
        yaw = self._refresh_rotation_pose()
        return self._rotate_to_yaw(yaw + math.radians(angle_deg))

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
            raise RuntimeError(f"找不到物件 {name}；中止指令。")
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
        self.turn(-90)

    def with_object_on_right(self, name: str):
        self.face(name)
        self.turn(90)

    def move_to_left(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_left_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_to_right(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_right_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def _footprint_goal_is_clear(self, pos, yaw, costs, meta, footprint_m,
                                 clearance_m):
        """Check an oriented convex polygon against lethal/unknown cells.

        Vertices are robot-frame meters relative to base_link. Check cell
        intersections including boundary cells and extra clearance.
        """
        import math
        footprint = np.asarray(footprint_m, dtype=float)
        if (footprint.ndim != 2 or footprint.shape[1] != 2 or len(footprint) < 3
                or not np.all(np.isfinite(footprint))):
            raise ValueError("footprint 必須是至少三個有效 [x,y] 頂點。")
        edges = np.roll(footprint, -1, axis=0)-footprint
        cross = edges[:,0]*np.roll(edges[:,1],-1)-edges[:,1]*np.roll(edges[:,0],-1)
        area = abs(np.sum(footprint[:,0]*np.roll(footprint[:,1],-1)
                          -footprint[:,1]*np.roll(footprint[:,0],-1)))/2
        if area < 1e-6 or (np.any(cross > 1e-9) and np.any(cross < -1e-9)):
            raise ValueError("footprint 必須是沿輪廓排序、面積大於零的凸多邊形。")
        x, y = self._grid_to_world(pos)
        c, sn = math.cos(yaw), math.sin(yaw)
        world = footprint @ np.array([[c, sn], [-sn, c]]) + [x, y]
        grid = (world - [meta.origin.position.x, meta.origin.position.y]) / meta.resolution
        height, width = costs.shape
        margin_cells = clearance_m/meta.resolution
        low = grid.min(axis=0)-margin_cells
        high = grid.max(axis=0)+margin_cells
        if np.any(low < 0) or high[0] >= width or high[1] >= height:
            return False
        lo = np.maximum(0, np.floor(low).astype(int)-1)
        hi = np.minimum([width-1,height-1], np.floor(high).astype(int)+1)
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        centers = np.column_stack([(xx+0.5).ravel(), (yy+0.5).ravel()])
        # Separating-axis intersection of convex footprint and every cell.
        # Expand each cell by the requested margin in x and y, conservatively
        # covering cell boundaries without requiring OpenCV rasterization.
        grid_edges = np.roll(grid,-1,axis=0)-grid
        axes = np.vstack([[1.,0.],[0.,1.],
                          np.column_stack([-grid_edges[:,1],grid_edges[:,0]])])
        overlaps = np.ones(len(centers), dtype=bool)
        for axis in axes:
            if np.linalg.norm(axis) < 1e-9:
                continue
            projection = grid @ axis
            centers_projection = centers @ axis
            half_extent = (0.5+margin_cells)*np.sum(np.abs(axis))
            overlaps &= ((centers_projection+half_extent >= projection.min()-1e-9)
                         & (centers_projection-half_extent <= projection.max()+1e-9))
        window = costs[lo[1]:hi[1]+1,lo[0]:hi[0]+1].ravel()
        ix = int(math.floor((x-meta.origin.position.x)/meta.resolution))
        iy = int(math.floor((y-meta.origin.position.y)/meta.resolution))
        if not (0 <= ix < width and 0 <= iy < height) or costs[iy, ix] >= 253:
            return False
        return not np.any(window[overlaps] >= 254)

    def move_in_between(self, name_a: str, name_b: str,
                        robot_radius_m=None, clearance_m=None,
                        footprint_m=None, search_band_m=0.60,
                        max_path_checks=24):
        """Find a reachable stopping pose between the selected objects.

        Sample goals near the object-center line and let Nav2 handle collision
        checking. Approximate planner endpoints are accepted. Size/clearance
        arguments are retained for API compatibility, without custom checks.
        """
        import math
        radius = self.between_robot_radius_m if robot_radius_m is None else robot_radius_m
        margin = self.between_clearance_m if clearance_m is None else clearance_m
        footprint = self.robot_footprint_m if footprint_m is None else footprint_m
        if (not all(math.isfinite(v) for v in (radius, margin, search_band_m))
                or radius <= 0 or margin < 0 or search_band_m <= 0
                or not isinstance(max_path_checks, int) or max_path_checks <= 0):
            raise ValueError("between 搜尋參數無效。")
        self._set_nav_curr_pose()
        a = np.asarray(self.get_pos(name_a), dtype=float).reshape(2)
        b = np.asarray(self.get_pos(name_b), dtype=float).reshape(2)
        delta = b-a
        length_px = float(np.linalg.norm(delta))
        if not np.all(np.isfinite([*a, *b])) or length_px < 1:
            raise RuntimeError("兩物件座標無效或重疊；無法定義 between。")
        direction = delta/length_px
        normal = np.array([-direction[1], direction[0]])
        midpoint = (a+b)/2
        # Sample at VLMap resolution, capped to a bounded number of points.
        count_along = min(201, max(3, int(math.ceil(length_px))+1))
        band_px = search_band_m/self.map.cs
        count_across = min(81, max(3, int(math.ceil(2*band_px))+1))
        curr_yaw = math.radians(self.curr_ang_deg_on_map)
        ax, ay = self._grid_to_world(a)
        bx, by = self._grid_to_world(b)
        # If the objects are opposite sides of a corridor, the corridor's
        # direction is perpendicular to their connecting line. Also retain
        # current heading; geometric alignment is only a candidate heuristic.
        tangent_yaw = math.atan2(by-ay, bx-ax)+math.pi/2
        yaws = [curr_yaw] if footprint is None else [curr_yaw, tangent_yaw, tangent_yaw+math.pi]
        candidates, seen = [], set()
        for t in np.linspace(0.05, 0.95, count_along):
            for offset in np.linspace(-band_px, band_px, count_across):
                point = a+t*delta+offset*normal
                pos = tuple(int(v) for v in np.rint(point))
                if pos in seen or not all(0 <= v < self.map.gs for v in pos):
                    continue
                seen.add(pos)
                for yaw in yaws:
                    distance_mid = np.linalg.norm(np.asarray(pos)-midpoint)*self.map.cs
                    rotation = abs(math.atan2(math.sin(yaw-curr_yaw), math.cos(yaw-curr_yaw)))
                    score = distance_mid + 0.05*rotation
                    candidates.append((score, pos, yaw))
        print(f"between: 使用 Nav2 避障檢查，候選位姿={len(candidates)}", flush=True)
        if not candidates:
            raise RuntimeError("兩物件間沒有有效的網格候選點。")
        candidates.sort(key=lambda item:item[0])
        start = self.current_world_pose
        checks = 0
        for _, pos, yaw in candidates:
            # Limit planning calls; alternate footprint orientations may
            # need separate plans and are counted separately.
            if checks >= max_path_checks:
                break
            checks += 1
            goal = self._make_goal(pos)
            q = tf_transformations.quaternion_from_euler(0, 0, yaw)
            (goal.pose.orientation.x, goal.pose.orientation.y,
             goal.pose.orientation.z, goal.pose.orientation.w) = q
            path = self.navigator.getPath(start, goal, use_start=True)
            if path is None or not path.poses:
                print(f"between 候選 {checks}: 停靠點={pos}，"
                      f"朝向={math.degrees(yaw):.1f}°，planner 未回傳有效路徑。",
                      flush=True)
                continue
            end_pose = path.poses[-1].pose
            end = end_pose.position
            end_error = math.hypot(end.x-goal.pose.position.x,
                                   end.y-goal.pose.position.y)
            # Use the endpoint returned by Nav2; retain numeric validity checks.
            end_pos = np.array([self.map.gs/2-end.y/self.map.cs,
                                self.map.gs/2+end.x/self.map.cs])
            end_q = end_pose.orientation
            quaternion = np.array([end_q.x, end_q.y, end_q.z, end_q.w])
            if (not np.all(np.isfinite(end_pos)) or
                    not np.all(np.isfinite(quaternion)) or
                    np.linalg.norm(quaternion) < 1e-6):
                print(f"between 候選 {checks}: 路徑終點位姿無效。", flush=True)
                continue
            quaternion /= np.linalg.norm(quaternion)
            _, _, end_yaw = tf_transformations.euler_from_quaternion(quaternion)
            goal.pose.position.x = float(end.x)
            goal.pose.position.y = float(end.y)
            (goal.pose.orientation.x, goal.pose.orientation.y,
             goal.pose.orientation.z, goal.pose.orientation.w) = quaternion
            print(f"between 採用 Nav2 路徑終點=({end.x:.3f}, {end.y:.3f})m，"
                  f"偏差={end_error:.3f}m，朝向={math.degrees(end_yaw):.1f}°",
                  flush=True)
            # Do not force yaw=0: turning across a narrow aisle can collide.
            result = self._navigate_and_wait(goal)
            self._set_nav_curr_pose()
            return result
        raise RuntimeError(f"{checks} 個候選位姿皆無可接受路徑；"
                           "請查看上述規劃失敗訊息與 Nav2 planner 日誌。")

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
        """Face the nearest object's map position using ROS world-frame yaw."""
        import math

        self._refresh_rotation_pose()
        target_pos = self.get_pos(name)
        target_x, target_y = self._grid_to_world(target_pos)
        current = self.current_world_pose.pose.position
        dx, dy = target_x - current.x, target_y - current.y
        if not all(math.isfinite(v) for v in (dx, dy)):
            raise RuntimeError("目標方向座標無效；中止指令。")
        if math.hypot(dx, dy) < 1e-6:
            raise RuntimeError("目標與機器人位置重疊；無法定義面向方向。")

        # ROS yaw: +X is zero, counterclockwise is positive. Use the
        # unrounded localization position for both bearing and stopping pose.
        yaw = math.atan2(dy, dx)
        print(f"面向 {name}，目標朝向: {math.degrees(yaw):.2f} 度", flush=True)
        return self._rotate_to_yaw(yaw)

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

    def _read_global_costmap(self):
        """Nav2 uint8 Costmap: 254 lethal, 255 unknown; not OccupancyGrid."""
        import math
        msg = self.navigator.getGlobalCostmap()
        if msg is None or msg.header.frame_id != "map":
            raise RuntimeError("無法取得 map 座標系的 Nav2 global costmap。")
        meta = msg.metadata
        if meta.resolution <= 0 or meta.size_x <= 0 or meta.size_y <= 0:
            raise RuntimeError("Nav2 costmap 尺寸或解析度無效。")
        q = meta.origin.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        if abs(yaw) > 1e-6:
            raise RuntimeError("此版本僅支援未旋轉的 Nav2 costmap origin。")
        costs = np.asarray(msg.data, dtype=np.uint8).reshape(meta.size_y, meta.size_x)
        return costs, meta

    def _costmap_goal_is_clear(self, pos, costs, meta, radius_m,
                               clearance_m, max_center_cost):
        """Conservative disk enclosing the robot, plus clearance.

        Treat unknown/outside-map cells as blocked. Disk checks lethal/unknown
        cells; center also respects inscribed and high inflation costs.
        """
        import math
        x, y = self._grid_to_world(pos)
        res = meta.resolution
        cx = (x - meta.origin.position.x) / res
        cy = (y - meta.origin.position.y) / res
        ix, iy = math.floor(cx), math.floor(cy)
        height, width = costs.shape
        radius_cells = (radius_m + clearance_m) / res
        if (cx - radius_cells < 0 or cy - radius_cells < 0 or
                cx + radius_cells >= width or cy + radius_cells >= height):
            return False
        if int(costs[iy, ix]) > max_center_cost:
            return False
        # Include any cell intersecting the disk using half-cell diagonal.
        extent = int(math.ceil(radius_cells + 1))
        x0, x1 = max(0, ix-extent), min(width, ix+extent+1)
        y0, y1 = max(0, iy-extent), min(height, iy+extent+1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = ((xx + 0.5 - cx)**2 + (yy + 0.5 - cy)**2
                <= (radius_cells + math.sqrt(0.5))**2)
        return not np.any(costs[y0:y1, x0:x1][disk] >= 254)

    def move_to_object(self, name: str, safe_distance_m: float = 0.50,
                       distance_tolerance_m: float = 0.10,
                       robot_radius_m: float = 0.45,
                       clearance_m: float = 0.05,
                       max_path_checks: int = 12,
                       max_center_cost: int = 200):
        """Choose a clear, reachable stopping point at the requested distance.

        Distance is robot/base center to the nearest semantic object edge,
        not robot bumper distance or the object's intrinsic front direction.
        robot_radius_m must enclose the actual footprint including payload.
        VLMap grid and ROS map must be aligned as in _grid_to_world().
        """
        import math
        values = [safe_distance_m, distance_tolerance_m, robot_radius_m, clearance_m]
        if (not all(math.isfinite(v) for v in values) or safe_distance_m <= 0
                or robot_radius_m <= 0 or clearance_m < 0
                or not 0 < distance_tolerance_m < safe_distance_m
                or not isinstance(max_path_checks, int) or max_path_checks <= 0
                or not 0 <= max_center_cost < 253):
            raise ValueError("停靠距離、容許誤差或安全參數無效。")
        self._set_nav_curr_pose()
        curr = np.asarray(self.curr_pos_on_map, dtype=float)
        edge = self.map.get_nearest_pos(self.curr_pos_on_map, name)
        if edge is None or np.array_equal(np.asarray(edge), curr):
            raise RuntimeError(f"找不到可靠的物件邊緣：{name}；中止指令。")
        edge = np.asarray(edge, dtype=float).reshape(2)
        if not np.all(np.isfinite(edge)):
            raise RuntimeError("物件邊緣座標無效。")
        direction = curr - edge
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            raise RuntimeError("機器人位置與物件邊緣重疊。")
        approach = direction / norm
        preferred = edge + approach * safe_distance_m / self.map.cs
        costs, meta = self._read_global_costmap()
        candidates = []
        seen = set()
        # Three rings around the nearest edge; preserve the approach-side
        # half-plane rather than silently moving to the object's far side.
        radii = (safe_distance_m, safe_distance_m-distance_tolerance_m*0.8,
                 safe_distance_m+distance_tolerance_m*0.8)
        points = [curr, preferred]
        for radius in radii:
            for angle in np.linspace(0, 2*math.pi, 72, endpoint=False):
                unit = np.array([math.sin(angle), math.cos(angle)])
                if np.dot(unit, approach) >= -1e-9:
                    points.append(edge + unit * radius / self.map.cs)
        for point in points:
            pos = tuple(np.rint(point).astype(int))
            if pos in seen or not all(0 <= value < self.map.gs for value in pos):
                continue
            seen.add(pos)
            if not self._costmap_goal_is_clear(pos, costs, meta, robot_radius_m,
                                                clearance_m, max_center_cost):
                continue
            # Re-query the semantic boundary: a circle around one edge alone
            # would give the wrong distance for wide objects or other edges.
            nearest = self.map.get_nearest_pos(list(pos), name)
            if nearest is None or np.array_equal(np.asarray(nearest), pos):
                continue
            nearest = np.asarray(nearest, dtype=float).reshape(2)
            # Keep the selected local edge neighborhood to reduce switching
            # to a distant instance of the same category.
            if np.linalg.norm(nearest-edge)*self.map.cs > safe_distance_m + distance_tolerance_m:
                continue
            measured = np.linalg.norm(np.asarray(pos)-nearest)*self.map.cs
            if abs(measured-safe_distance_m) > distance_tolerance_m:
                continue
            score = np.linalg.norm(np.asarray(pos)-preferred)*self.map.cs
            candidates.append((score, pos, nearest, measured))
        if not candidates:
            raise RuntimeError(f"找不到 {name} 外 {safe_distance_m:.2f}±"
                               f"{distance_tolerance_m:.2f}m 的安全停靠點；中止指令。")
        candidates.sort(key=lambda item: item[0])
        start = self.current_world_pose
        for _, pos, face_pos, measured in candidates[:max_path_checks]:
            goal = self._make_goal(pos, face_pos)
            path = self.navigator.getPath(start, goal, use_start=True)
            if path is None or not path.poses:
                continue
            # Planner tolerance may return a path ending near, rather than
            # at, the requested goal. Reject larger endpoint deviations.
            end = path.poses[-1].pose.position
            endpoint_error = math.hypot(end.x-goal.pose.position.x,
                                        end.y-goal.pose.position.y)
            if endpoint_error > min(distance_tolerance_m/2, meta.resolution):
                continue
            # Refresh costmap before sending the actual navigation action.
            latest_costs, latest_meta = self._read_global_costmap()
            if not self._costmap_goal_is_clear(pos, latest_costs, latest_meta,
                                               robot_radius_m, clearance_m,
                                               max_center_cost):
                continue
            print(f"{name} 停靠點 {pos}，地圖邊緣距離 {measured:.2f}m", flush=True)
            result = self._navigate_and_wait(goal)
            # Nav2 success uses its configured goal tolerance. Independently
            # check actual localized distance before allowing the next step.
            self._set_nav_curr_pose()
            actual_edge = self.map.get_nearest_pos(self.curr_pos_on_map, name)
            if actual_edge is None:
                raise RuntimeError("抵達後無法驗證物件距離；中止後續動作。")
            ex, ey = self._grid_to_world(actual_edge)
            p = self.current_world_pose.pose.position
            actual_distance = math.hypot(p.x-ex, p.y-ey)
            if abs(actual_distance-safe_distance_m) > distance_tolerance_m:
                raise RuntimeError(f"抵達後定位距離 {actual_distance:.2f}m 不符合 "
                                   f"{safe_distance_m:.2f}±{distance_tolerance_m:.2f}m；"
                                   "中止後續動作。請檢查 Nav2 goal tolerance 與定位。")
            return result
        raise RuntimeError(f"前 {min(len(candidates), max_path_checks)} 個候選點"
                           "皆無可接受路徑；中止指令。可增加 max_path_checks。")

    def move_forward(self, meters: float):
        self._set_nav_curr_pose()
        pos = self.map.get_forward_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, meters)
        self.move_to(pos)
