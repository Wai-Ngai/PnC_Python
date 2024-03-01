import math
import numpy as np


class PurePursuitController:
    def calc_target_index(self, vehicle_state, reference_path, ld):
        # 找到规划轨迹距离自车最近的点
        distances = []
        for xy in reference_path:
            dis = np.linalg.norm(vehicle_state - xy)
            distances.append(dis)

        min_index = np.argmin(distances)

        # 根据预瞄距离，计算预瞄点
        delta_l = np.linalg.norm(reference_path[min_index] - vehicle_state)
        while ld > delta_l and (min_index + 1) < len(reference_path):
            delta_l = np.linalg.norm(reference_path[min_index + 1] - vehicle_state)
            min_index += 1
        return min_index

    def control(self, vehicle_state, current_ref_point, L, ld, psi):
        alpha = math.atan2(current_ref_point[1] - vehicle_state[1], current_ref_point[0] - vehicle_state[0]) - psi
        delta_f = math.atan2(2 * L * np.sin(alpha), ld)
        return delta_f
