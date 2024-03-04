import math

import numpy as np


class StanleyController:

    def calc_target_index(self, vehicle_state, reference_path):
        """
            计算参考轨迹上距离自车最近的点
        Args:
            vehicle_state:
            reference_path:

        Returns:

        """

        distances = []
        for xy in reference_path:
            dis = np.linalg.norm(vehicle_state - xy)
            distances.append(dis)

        min_index = np.argmin(distances)
        return min_index

    def normalize(self, angle):
        if angle > np.pi:
            angle = angle - 2 * np.pi
        elif angle < -np.pi:
            angle = angle + 2 * np.pi

        return angle

    def control(self, vehicle_state, reference_path, reference_path_psi,k):
        # 计算参考点信息
        current_target_index = self.calc_target_index(vehicle_state[0:2], reference_path)
        if current_target_index >= len(reference_path):
            current_target_index = len(reference_path) - 1
        current_ref_point = reference_path[current_target_index]
        psi_t = reference_path_psi[current_target_index]

        # 计算横向误差
        if (vehicle_state[0] - current_ref_point[0]) * math.sin(psi_t) - (vehicle_state[1] - current_ref_point[1])*math.cos(psi_t) > 0:
            error_y = np.linalg.norm(vehicle_state[0:2]-current_ref_point)
        else:
            error_y = -np.linalg.norm(vehicle_state[0:2] - current_ref_point)

        psi = vehicle_state[2]
        velocity = vehicle_state[3]

        theta_e = psi_t - psi                      # 航向误差引起的转角
        delta_e = math.atan2(k*error_y,velocity)   # 横向误差引起的转角
        delta_f = self.normalize(theta_e + delta_e)

        return delta_f,current_target_index