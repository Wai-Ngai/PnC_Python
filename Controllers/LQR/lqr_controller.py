import numpy as np


class LQRController:

    def __init__(self, max_iteration_num, eps):
        self.max_iteration_num = max_iteration_num
        self.eps = eps

    def calc_ricatti(self, A, B, Q, R):
        """
            解代数里卡提方程
        Args:
            A:  状态矩阵A
            B:  状态矩阵B
            Q:  Q为半正定的状态加权矩阵，通常为对角矩阵；Q的元素变大意味着希望跟踪偏差能够快速趋近于0
            R:  R为正定的控制加权矩阵，R矩阵元素变大意味着希望控制输入能够尽可能小

        Returns:

        """
        # 设置迭代初始值
        Qf = Q
        P = Qf
        # 循环迭代
        for t in range(self.max_iteration_num):
            P_new = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            if (abs(P_new - P).max() < self.eps):
                break
            P = P_new
        return P

    def control(self, vehicle_state, reference_path, min_index, A, B, Q, R):
        # 误差状态：位置x误差，位置y误差，位置heading误差
        X = vehicle_state[0:3] - reference_path[min_index, 0:3]

        P = self.calc_ricatti(A, B, Q, R)
        k = -np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
        u = k @ X
        u_star = u  # u_star = [[v-ref_v,delta-ref_delta]]

        return u_star[1]
