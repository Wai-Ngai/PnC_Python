import math
import numpy as np


class KinematicModel1:
    """
        以车辆质心为中心的车辆运动学模型
    """

    def __init__(self, x, y, psi, v, lr, lf, dt):
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v
        self.lr = lr
        self.lf = lf
        self.dt = dt

    def updta_state(self, a, delta_f, delta_r):
        beta = math.atan2((self.lr * math.tan(delta_f) + self.lf * math.tan(delta_r)) / (self.lf + self.lr))
        self.x = self.x + self.v * math.cos(self.psi + beta) * self.dt
        self.y = self.y + self.v * math.sin(self.psi + beta) * self.dt
        self.psi = self.psi + self.v * math.cos(beta) * (math.tan(delta_f) - math.tan(delta_r)) / (
                self.lr + self.lr) * self.dt
        self.v = self.v + a * self.dt

    def get_state(self):
        return self.x, self.y, self.psi, self.v


class KinematicModel2:
    """
        前轮转向车辆运动学模型
    """

    def __init__(self, x, y, psi, v, lr, lf, dt):
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v
        self.lr = lr
        self.lf = lf
        self.dt = dt

    def updta_state(self, a, delta_f):
        beta = math.atan2((self.lr * math.tan(delta_f)) / (self.lf + self.lf))
        self.x = self.x + self.v * math.cos(self.psi + beta) * self.dt
        self.y = self.y + self.v * math.sin(self.psi + beta) * self.dt
        self.psi = self.psi + self.v * math.sin(beta) / self.lr * self.dt
        self.v = self.v + a * self.dt

    def get_state(self):
        return self.x, self.y, self.psi, self.v


class KinematicModel3:
    """
        以后轴中心为车辆中心的车辆运动学模型
    """

    def __init__(self, x, y, psi, v, L, dt):
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v
        self.L = L
        self.dt = dt

    def updta_state(self, delta_f, a):
        self.x = self.x + self.v * math.cos(self.psi) * self.dt
        self.y = self.y + self.v * math.sin(self.psi) * self.dt
        self.psi = self.psi + self.v * math.tan(delta_f) / self.L * self.dt
        self.v = self.v + a * self.dt

    def get_state(self):
        return self.x, self.y, self.psi, self.v

    def state_space(self, ref_delta, ref_yaw):
        """
            将模型离散化后的状态空间表达式
        Args:
            ref_delta:参考轨迹转角？
            ref_yaw:参考轨迹航向角

        Returns:

        """
        A = np.array([[1.0, 0.0, -self.v * math.sin(ref_yaw) * self.dt],
                      [0.0, 1.0, self.v * math.cos(ref_yaw) * self.dt],
                      [0.0, 0.0, 1.0]])
        B = np.array([[self.dt * math.cos(ref_yaw), 0],
                      [self.dt * math.sin(ref_yaw), 0],
                      [self.dt * math.tan(ref_delta) / self.L,
                       self.v * self.dt / (self.L * math.cos(ref_delta) ** 2)]])

        return A, B


class LateralErrorModel:
    """
        车辆横向误差运动学模型
    """

    def __init__(self, m, Vx, C_alpha_f, C_alpha_r, lf, lr, Iz, g):
        self.m = m
        self.Vx = Vx
        self.C_alpha_f = C_alpha_f
        self.C_alpha_r = C_alpha_r
        self.lf = lf
        self.lr = lr
        self.Iz = Iz
        self.g = g

    def generate_state_space(self):
        A = np.array([[0., 1., 0., 0],
                      [0., - (2 * self.C_alpha_f + 2 * self.C_alpha_r) / (self.m * self.Vx),
                       (2 * self.C_alpha_f + 2 * self.C_alpha_r) / self.m,
                       (-2 * self.C_alpha_f * self.lf + 2 * self.C_alpha_r * self.lr) / (self.m * self.Vx)],
                      [0., 0., 0., 1.],
                      [0, -(2 * self.lf * self.C_alpha_f - 2 * self.lr * self.C_alpha_r) / (self.Iz * self.Vx),
                       (2 * self.lf * self.C_alpha_f - 2 * self.lr * self.C_alpha_r) / (self.Iz),
                       -(2 * self.lf ** 2 * self.C_alpha_f + 2 * self.lr ** 2 * self.C_alpha_r) / (self.Iz * self.Vx)]])

        B = np.array([[0],
                      [2 * self.C_alpha_f / self.m],
                      [0],
                      [2 * self.lf * self.C_alpha_f / self.Iz]])

        C = np.array([[0],
                      [(-2 * self.lf * self.C_alpha_f + 2 * self.lr * self.C_alpha_r) / (self.m * self.Vx) - self.Vx],
                      [0],
                      [-(2 * self.lf ** 2 * self.C_alpha_f + 2 * self.lr ** 2 * self.C_alpha_r) / (self.Iz * self.Vx)]])
        D = np.array([[0],
                      [self.g],
                      [0],
                      [0]])

        return A, B, C, D

    def compute_state_derivative(self, state, delta, psi_des, phi):
        A, B, C, D = self.generate_state_space()
        state_dot = np.dot(A, state) + np.dot(B, delta) + np.dot(C, psi_des) + np.dot(D, np.sin(phi))
        return state_dot[:, 0]

    def discrete_state_space(self, dt):
        A, B, C, D = self.generate_state_space()
        I = np.eye(4)
        A_bar = I + A * dt
        B_bar = B * dt
        return A_bar, B_bar
