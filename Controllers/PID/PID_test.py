import math
import pid_controller as pid
from PID_controller import PID_posi_2
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from scipy.spatial import KDTree


class KinematicModel:
    def __init__(self, x, y, psi, v, L, dt):
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v
        self.L = L
        self.dt = dt

    def update(self, a, delta_f):
        self.x = self.x + self.v * math.cos(self.psi) * self.dt
        self.y = self.y + self.v * math.sin(self.psi) * self.dt
        self.psi = self.psi + self.v / self.L * math.tan(delta_f) * self.dt
        self.v = self.v + a * self.dt

    def get_state(self):
        return [self.x, self.y, self.psi, self.v]


def main():
    # set reference trajectory
    ref_path = np.zeros((1000,2))
    ref_path[:,0] = np.linspace(0,100,1000)
    ref_path[:,1] = 2 * np.sin(ref_path[:,0] / 3.0)
    ref_tree = KDTree(ref_path)

    # 设置初始状态
    model = KinematicModel(0,-1,0.5,2,2,0.1)
    x_ = []
    y_ = []
    PID = pid.PIDController(2.0,0.01,30,upper=np.pi/6, lower=-np.pi/6)
    # PID = PID_posi_2(k=[2, 0.01, 30], target=0, upper=np.pi / 6, lower=-np.pi / 6)

    fig = plt.figure(1)
    camera = Camera(fig)

    for i in range(550):
        robot_state = np.zeros(2)
        robot_state[0] = model.x
        robot_state[1] = model.y
        distance,index =ref_tree.query(robot_state)

        alpha = math.atan2(ref_path[index,1] - robot_state[1], ref_path[index,0]- robot_state[0])
        ld = np.linalg.norm(ref_path[index] - robot_state)

        theta = alpha- model.psi
        err_y = ld * math.sin(theta)

        delta_f = PID.calculate(err_y,0.01)
        # delta_f = PID.cal_output(err_y)
        print(delta_f)
        model.update(0,delta_f)

        x_.append(model.x)
        y_.append(model.y)

        # 画图
        plt.cla()
        plt.plot(ref_path[:,0],ref_path[:,1],'-.b',linewidth=1.0)
        plt.plot(x_,y_,'r',label='trajectory')
        plt.plot(ref_path[index,0],ref_path[index,1],'go',label='target')

        plt.grid(True)
        plt.legend(loc='right')
        plt.pause(0.01)
        camera.snap()

    animation = camera.animate()
    animation.save('trajectory1.gif')

if __name__ == '__main__':
    main()
