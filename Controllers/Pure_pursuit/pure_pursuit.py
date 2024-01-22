import math
import numpy as np


class PurePursuit():
    def __init__(self):
        pass

    def cal_target_index(self, robot_state, refer_path, ld):
        # 找到规划轨迹距离自车最近的点
        distances = []
        for xy in refer_path:
            dis = np.linalg.norm(robot_state - xy)
            distances.append(dis)

        min_index = np.argmin(distances)

        # 根据预瞄距离，计算预瞄点
        delta_l = np.linalg.norm(refer_path[min_index] - robot_state)
        while ld > delta_l and (min_index + 1) < len(refer_path):
            delta_l = np.linalg.norm(refer_path[min_index + 1] - robot_state)
            min_index += 1
        return min_index

    def control(self, robot_state, current_ref_point, L, ld, psi):
        alpha = math.atan2(current_ref_point[1] - robot_state[1], current_ref_point[0] - robot_state[0]) - psi
        delta = math.atan2(2 * L * np.sin(alpha), ld)
        return delta
