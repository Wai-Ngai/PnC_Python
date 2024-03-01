from celluloid import Camera
from pure_pursuit_controller import PurePursuitController
from DynamicsAndKinematicModel.kinematic_model import KinematicModel3
import numpy as np
import matplotlib.pyplot as plt
import math

def test():
    # 整车参数
    L = 2        # 车辆轴距
    V = 2        # 初始速度
    x_0 = 0      # 初始x
    y_0 = -3     # 初始y
    psi_0 = 0    # 初始航向角
    dt = 0.1     # 时间间隔
    k = 0.1      # 预瞄距离系数
    c = 2        # 预瞄距离常数

    # 初始化参考轨迹
    reference_path = np.zeros((1000,2))
    reference_path[:,0] = np.linspace(0,100,1000)
    reference_path[:,1] = 2*np.sin(reference_path[:,0]/3.0) + 2.5 * np.cos(reference_path[:,0]/2.0)

    model = KinematicModel3(x_0, y_0, psi_0, V,L,dt)
    vehicle_x = []
    vehicle_y = []

    fig = plt.figure()
    camera = Camera(fig)
    pure_pursuit_controller = PurePursuitController()

    for i in range(600):
        vehicle_state = np.zeros(2)
        vehicle_state[0] = model.x
        vehicle_state[1] = model.y

        ld = k * model.v + c
        index = pure_pursuit_controller.calc_target_index(vehicle_state,reference_path,ld)
        delta_f = pure_pursuit_controller.control(vehicle_state,reference_path[index],L,ld,model.psi)

        model.updta_state(delta_f,0)
        vehicle_x.append(model.x)
        vehicle_y.append(model.y)

        plt.cla()
        plt.plot(reference_path[:,0],reference_path[:,1],'-.b',linewidth=1.0)
        plt.plot(vehicle_x,vehicle_y,'-r',linewidth=1.0,label='trajectory')
        plt.plot(reference_path[index,0],reference_path[index,1],"go",label='target')

        plt.grid(True)
        plt.pause(0.001)

        camera.snap()
    animation = camera.animate()
    animation.save('trajectory.gif')


if __name__ == '__main__':
    test()