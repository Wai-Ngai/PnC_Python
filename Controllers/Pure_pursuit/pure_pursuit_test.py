from celluloid import Camera
from pure_pursuit import PurePursuit
import numpy as np
import matplotlib.pyplot as plt
import math

L = 2
v = 2
x_0 = 0
y_0 = -3
psi_0 = 0
dt = 0.1
k = 0.1
c = 2


class KinematicModel:
    def __init__(self, x, y, psi, v, L, dt):
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v
        self.L = L
        self.dt = dt

    def update(self, a, delta_f):
        self.x += self.v * math.cos(self.psi) * self.dt
        self.y += self.v * math.sin(self.psi) * self.dt
        self.psi += self.v / self.L * math.tan(delta_f) * self.dt
        self.v += a * self.dt

    def get_state(self):
        return self.x, self.y, self.psi, self.v


def main():
    ref_path = np.zeros((1000, 2))
    ref_path[:, 0] = np.linspace(0, 100, 1000)
    ref_path[:, 1] = 2 * np.sin(ref_path[:, 0] / 3.0) + 2.5 * np.cos(ref_path[:, 0] / 2.0)

    model = KinematicModel(x_0, y_0, psi_0, v, L, dt)
    x_ = []
    y_ = []
    fig = plt.figure()
    camera = Camera(fig)

    pure_pursuit = PurePursuit()
    for i in range(600):
        robot_state = np.zeros(2)
        robot_state[0] = model.x
        robot_state[1] = model.y

        ld = k * model.v + c
        index = pure_pursuit.cal_target_index(robot_state, ref_path, ld)
        delta = pure_pursuit.control(robot_state, ref_path[index], L, ld, model.psi)

        model.update(0, delta)
        x_.append(model.x)
        y_.append(model.y)

        plt.cla()
        plt.plot(ref_path[:, 0], ref_path[:,1], '-.b', linewidth=1.0)
        plt.plot(x_, y_, '-r', label="trajectory", linewidth=1.0)
        plt.plot(ref_path[index, 0], ref_path[index, 1], "go",label="target")

        plt.grid(True)
        plt.pause(0.001)

        camera.snap()
    animation = camera.animate()
    animation.save('trajectory1.gif')


if __name__ == "__main__":
    main()
