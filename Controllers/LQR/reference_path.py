import numpy as np
import math


class ReferencePath:
    """
        提供一条用于跟踪的参考路径
    """

    def __init__(self):
        # 路径初始化
        self.reference_path = np.zeros((1000, 4))
        self.reference_path[:, 0] = np.linspace(0, 100, 1000)
        self.reference_path[:, 1] = 2 * np.sin(self.reference_path[:, 0] / 3.0) + 2.5 * np.cos(
            self.reference_path[:, 0] / 2.0)

        # 使用差分计算路径点的一阶导和二阶导，从而得到路径的heading和Kappa
        for i in range(len(self.reference_path)):
            if i == 0:
                dx = self.reference_path[i + 1, 0] - self.reference_path[i, 0]
                dy = self.reference_path[i + 1, 1] - self.reference_path[i, 1]
                ddx = self.reference_path[i + 2, 0] - 2 * self.reference_path[i + 1, 0] + self.reference_path[i, 0]
                ddy = self.reference_path[i + 2, 1] - 2 * self.reference_path[i + 1, 1] + self.reference_path[i, 1]
            elif i == len(self.reference_path) - 1:
                dx = self.reference_path[i, 0] - self.reference_path[i - 1, 0]
                dy = self.reference_path[i, 1] - self.reference_path[i - 1, 1]
                ddx = self.reference_path[i, 0] - 2 * self.reference_path[i - 1, 0] + self.reference_path[i - 2, 0]
                ddy = self.reference_path[i, 1] - 2 * self.reference_path[i - 1, 1] + self.reference_path[i - 2, 0]
            else:
                dx = self.reference_path[i + 1, 0] - self.reference_path[i, 0]
                dy = self.reference_path[i + 1, 1] - self.reference_path[i, 1]
                ddx = self.reference_path[i + 1, 0] - 2 * self.reference_path[i, 0] + self.reference_path[i - 1, 0]
                ddy = self.reference_path[i + 1, 1] - 2 * self.reference_path[i, 1] + self.reference_path[i - 1, 1]

            self.reference_path[i, 2] = math.atan2(dy, dx)  # heading
            self.reference_path[i, 3] = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))  # kappa

    def normalize_angle(self, angle):
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def calc_track_error(self, x, y):
        """
            根据车辆当前的位置计算跟踪误差
        Args:
            x: 车辆当前的位置x
            y: 车辆当前的位置y

        Returns:

        """
        # 寻找参考轨迹最近的点
        d_x = [self.reference_path[i, 0] - x for i in range(len(self.reference_path))]
        d_y = [self.reference_path[i, 1] - y for i in range(len(self.reference_path))]
        distances = [np.sqrt(d_x[i] ** 2 + d_y[i] ** 2) for i in range(len(d_x))]
        min_index = np.argmin(distances)  # 最近目标点的index

        # 最近点的信息
        heading = self.reference_path[min_index, 2]
        kappa = self.reference_path[min_index, 3]
        error_y = distances[min_index]

        angle = self.normalize_angle(heading - math.atan2(d_y[min_index],d_x[min_index]))
        if angle < 0:
            error_y *= -1
        return error_y, kappa,heading,min_index