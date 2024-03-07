import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

from Controllers.MPC.mpc_controller import MPCController
from DynamicsAndKinematicModel.kinematic_model import KinematicModel3
from Controllers.LQR.reference_path import ReferencePath

# MPC parameters
NX = 3  # state numbers x = x, y, yaw
NU = 2  # actuator numbers u = [v,delta]
Hp = 8  # predictive horizon
Hu = 8  # control horizon
R = np.diag([0.1, 0.1])  # input cost matrix
Rd = np.diag([0.1, 0.1])  # input difference cost matrix
Q = np.diag([1, 1, 1])  # state cost matrix
Qf = Q  # state final matrix

# 整车参数
dt = 0.1  # 时间间隔，单位：s
L = 2  # 车辆轴距，单位：m,
v = 2  # 初始速度
x_0 = 0  # 初始x
y_0 = -3  # 初始y
psi_0 = 0  # 初始航向角

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_VELOCITY = 2.0  # maximum speed [m/s]


def main():
    path = ReferencePath()
    goal = path.reference_path[-1, 0:2]  # 取最后一行的xy

    model = KinematicModel3(x_0, y_0, psi_0, v, L, dt)
    mpc = MPCController(Hp, Hu, NX, NU, R, Q, Qf, dt)

    vehicle_x = []
    vehicle_y = []
    fig = plt.figure(1)
    camera = Camera(fig)

    for i in range(700):
        vehicle_state = np.zeros(4)
        vehicle_state[0] = model.x
        vehicle_state[1] = model.y
        vehicle_state[2] = model.psi
        vehicle_state[3] = model.v
        X0 = vehicle_state[0:3]  # 整车的初始状态，用于状态方程初始状态计算

        # 计算规划轨迹中距离自车最近点信息
        error_y, kappa, heading, min_index = path.calc_track_error(model.x, model.y)
        # 计算参考轨迹和参考控制量，用于状态空间方程的计算
        x_ref, u_ref = mpc.calc_ref_trajectory(vehicle_state, path.reference_path, min_index, kappa, L)
        # MPC算法计算转角
        optimal_x, optimal_y, optimal_yaw, optimal_v, optimal_delta = mpc.control(model, x_ref, u_ref, X0, MAX_VELOCITY,
                                                                                  MAX_STEER)
        # 取出转角中第一个控制量，施加在模型上
        model.updta_state(optimal_delta[0], 0)
        vehicle_x.append(model.x)
        vehicle_y.append(model.y)

        # 画图
        plt.cla()
        plt.plot(path.reference_path[:, 0], path.reference_path[:, 1], "-b", linewidth=1.0, label="Reference")
        plt.plot(path.reference_path[min_index, 0], path.reference_path[min_index, 1], "go", label="target")
        plt.plot(vehicle_x, vehicle_y, "-r", linewidth=1.0, label="Vehicle")
        plt.grid(True)
        plt.pause(0.001)
        camera.snap()

        # 判断是否达到最后一个点
        if np.linalg.norm(vehicle_state[0:2] - goal) <= 0.1:
            print("Goal reached")
            break
    animation = camera.animate()
    animation.save("trajectory1.gif")

if __name__ == '__main__':
    main()
