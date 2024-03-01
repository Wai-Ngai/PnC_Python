import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import KDTree
from celluloid import Camera
from pid_controller import PIDController1
from DynamicsAndKinematicModel.kinematic_model import KinematicModel3


def calc_target_index(vehicle_state, reference_path):
    """
        计算规划轨迹中的最近点
    Args:
        vehicle_state: 当前车辆位置
        reference_path: 参考轨迹

    Returns:

    """

    dists = []
    for xy in reference_path:
        dis = np.linalg.norm(vehicle_state - xy)
        dists.append(dis)

    min_index = np.argmin(dists)
    return min_index


def main():
    # 初始化参考轨迹
    reference_path = np.zeros((1000, 2))
    reference_path[:, 0] = np.linspace(0, 100, 1000)
    reference_path[:, 1] = 2 * np.sin(reference_path[:, 0] / 3.0)

    PID = PIDController1(2, 0.01, 30, 0, upper=np.pi / 6, lower=-np.pi / 6)
    model = KinematicModel3(0, -1, 0.5, 2, 2, 0.1)
    vehicle_x = []
    vehicle_y = []

    fig = plt.figure(1)
    camera = Camera(fig)

    for i in range(500):
        vehicle_state = np.zeros(2)
        vehicle_state[0] = model.x
        vehicle_state[1] = model.y

        # 寻找参考轨迹上距离自车最近的点
        index = calc_target_index(vehicle_state, reference_path)
        alpha = math.atan2(reference_path[index, 1] - vehicle_state[1], reference_path[index, 0] - vehicle_state[0])
        ld = np.linalg.norm(reference_path[index] - vehicle_state)

        theta = alpha - model.psi
        error_y = ld * math.sin(theta)

        delta_f = PID.calculate(error_y, 0.01)
        model.updta_state(delta_f, 0)

        vehicle_x.append(model.x)
        vehicle_y.append(model.y)

        # 画图
        plt.cla()
        plt.plot(reference_path[:, 0], reference_path[:, 1], '-.b', linewidth=1.0)
        plt.plot(vehicle_x, vehicle_y, '-r', label="trajectory")
        plt.plot(reference_path[index, 0], reference_path[index, 1], 'go', label="target")

        plt.grid(True)
        plt.legend()
        plt.pause(0.01)
        camera.snap()

    animation = camera.animate()
    animation.save('trajectory.gif')


if __name__ == '__main__':
    main()
