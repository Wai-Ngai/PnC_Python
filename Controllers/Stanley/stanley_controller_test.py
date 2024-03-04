import numpy as np
import matplotlib.pyplot as plt
import math
from celluloid import Camera
from DynamicsAndKinematicModel.kinematic_model import KinematicModel3
from stanley_controller import StanleyController

k = 0.2  # 增益系数
dt = 0.1  # 时间间隔，单位：s

L = 2  # 车辆轴距，单位：m
v = 2  # 初始速度
x_0 = 0  # 初始x
y_0 = -3  # 初始y
psi_0 = 0  # 初始航向角


def main():
    # 初始化跟踪轨迹
    reference_path = np.zeros((1000, 2))
    reference_path[:, 0] = np.linspace(0, 100, 1000)
    reference_path[:, 1] = 2 * np.sin(reference_path[:, 0] / 3.0) + 2.5 * np.cos(reference_path[:, 0] / 2.0)
    reference_path_psi = [
        math.atan2(reference_path[i + 1, 1] - reference_path[i, 1], reference_path[i + 1, 0] - reference_path[i, 0]) for
        i in range(len(reference_path) - 1)]  # 列表推导式

    # 运动学模型
    model = KinematicModel3(x_0, y_0, psi_0, v, L, dt)

    stanley = StanleyController()
    vehicle_x = []
    vehicle_y = []
    fig = plt.figure(1)
    camera = Camera(fig)

    for i in range(500):
        # 获取车辆状态
        vehicle_state = np.zeros(4)
        vehicle_state[0] = model.x
        vehicle_state[1] = model.y
        vehicle_state[2] = model.psi
        vehicle_state[3] = model.v

        delta_f,index = stanley.control(vehicle_state,reference_path,reference_path_psi,k)
        model.updta_state(delta_f,0)
        vehicle_x.append(model.x)
        vehicle_y.append(model.y)

        # 显示动图
        plt.cla()
        plt.plot(reference_path[:,0],reference_path[:,1],'-.b',linewidth=1.0)
        plt.plot(vehicle_x,vehicle_y,'-r',label="trajectory")
        plt.plot(reference_path[index,0],reference_path[index,1],'go',label="target")
        plt.grid(True)
        plt.pause(0.001)

    animation= camera.animate()
    animation.save("trajectory1.gif")




if __name__ == '__main__':
    main()
