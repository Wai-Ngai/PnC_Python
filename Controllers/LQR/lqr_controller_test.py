import numpy as np
import matplotlib.pyplot as plt
import math
from celluloid import Camera
from DynamicsAndKinematicModel.kinematic_model import KinematicModel3
from reference_path import ReferencePath
from lqr_controller import LQRController

# 整车参数
dt = 0.1
L = 2  # 整车轴距
v = 2  # 初始速度
x_0 = 0
y_0 = -3
psi_0 = 0

# LQR 参数
max_iteration_num = 100  # 迭代范围
eps = 1e-4  # 迭代精度
Q = np.eye(3) * 3
R = np.eye(2) * 2


def main():
    # 初始化跟踪轨迹
    path = ReferencePath()
    goal = path.reference_path[-1, 0:2]

    # 运动学模型
    model = KinematicModel3(x_0, y_0, psi_0, v, L, dt)
    # 初始化LQR控制器
    lqr = LQRController(max_iteration_num, eps)

    # 画图用
    vehicle_x = []
    vehicle_y = []
    fig = plt.figure(1)
    camera = Camera(fig)

    for i in range(700):
        # 根据运动学模型得到车辆当前位姿
        vehicle_state = np.zeros(4)
        vehicle_state[0] = model.x
        vehicle_state[1] = model.y
        vehicle_state[2] = model.psi
        vehicle_state[3] = model.v

        # 计算规划轨迹中距离自车最近点的信息
        error_y, kappa, heading, min_index = path.calc_track_error(vehicle_state[0], vehicle_state[1])

        # 计算离散模型的矩阵
        delta_ref = math.atan2(kappa * L, 1)  # tanδ = L / R
        A, B = model.state_space(delta_ref, heading)

        # lqr计算转角差
        delta = lqr.control(vehicle_state, path.reference_path, min_index, A, B, Q, R)
        delta_f = delta + delta_ref

        # 更新状态
        model.updta_state(delta_f, 0)

        # 画图
        vehicle_x.append(model.x)
        vehicle_y.append(model.y)

        plt.cla()
        plt.plot(path.reference_path[:, 0], path.reference_path[:, 1], "-b", linewidth=1.0, label="Reference")
        plt.plot(vehicle_x, vehicle_y, "-r", linewidth=1.0, label="Vehicle")
        plt.plot(path.reference_path[min_index, 0], path.reference_path[min_index, 1], "go", linewidth=1.0, label="target")

        plt.grid(True)
        plt.pause(0.01)
        camera.snap()

        # 判断是否到达最后一个点
        if np.linalg.norm(vehicle_state[0:2] - goal) <= 0.1:
            print("Goal reached")
            break
    animation = camera.animate()
    animation.save("trajectory1.gif")


if __name__ == '__main__':
    main()
