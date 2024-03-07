import numpy as np
import math
import cvxpy


class MPCController:
    """
    The linear MPC controller
    """

    def __init__(self, Hp, Hu, NX, NU, R, Q, Qf, dt):
        self.Hp = Hp  # prediction horizon
        self.Hu = Hu  # control horizon
        self.NX = NX  # state numbers
        self.NU = NU  # actuator numbers
        self.R = R  # input cost matrix
        self.Q = Q  # state cost matrix
        self.Qf = Qf  # final state matrix
        self.dt = dt

    def matrix_to_ndarray(self, x):
        """
            将数组 x 转换为 NumPy 数组，并且将其展开为一维数组
        Args:
            x:

        Returns:

        """
        return np.array(x).flatten()

    def calc_ref_trajectory(self, vehicle_state, reference_path, min_index, kappa, L, dl=1):
        x_ref = np.zeros((self.NX, self.Hp + 1))
        u_ref = np.zeros((self.NU, self.Hp))

        # 参考控制量
        delta_ref = math.atan2(kappa * L, 1)
        u_ref[0, :] = vehicle_state[3]  # v_ref
        u_ref[1, :] = delta_ref

        # 参考轨迹的起点
        x_ref[0, 0] = reference_path[min_index, 0]  # x
        x_ref[1, 0] = reference_path[min_index, 1]  # y
        x_ref[2, 0] = reference_path[min_index, 2]  # psi

        length = len(reference_path)
        s = 0.0
        for i in range(self.Hp + 1):
            s += abs(vehicle_state[3]) * self.dt  # 累计行驶距离
            index = int(round(s / dl))

            if (min_index + index) < length:
                x_ref[0, i] = reference_path[min_index + index, 0]
                x_ref[1, i] = reference_path[min_index + index, 1]
                x_ref[2, i] = reference_path[min_index + index, 2]
            else:
                x_ref[0, i] = reference_path[length - 1, 0]
                x_ref[1, i] = reference_path[length - 1, 1]
                x_ref[2, i] = reference_path[length - 1, 2]

        return x_ref, u_ref

    def control(self, model, x_ref, u_ref, X_0, MAX_VELOCITY, MAX_STEER_ANGLE):
        # 创建优化变量，用于cvxpy优化计算
        x = cvxpy.Variable((self.NX, self.Hp + 1))
        u = cvxpy.Variable((self.NU, self.Hp))

        # 计算代价函数和约束条件
        cost = 0.0
        constraints = []
        for i in range(self.Hp):
            cost += cvxpy.quad_form(u[:, i] - u_ref[:, i], self.R)  # uTRu
            if i != 0:
                cost += cvxpy.quad_form(x[:, i] - x_ref[:, i], self.Q)  # xTQx

            A, B = model.state_space(u_ref[1, i], x_ref[2, i])
            constraints += [x[:, i + 1] - x_ref[:, i + 1] == A @ (x[:, i] - x_ref[:, i]) + B @ (u[:, i] - u_ref[:, i])]

        cost += cvxpy.quad_form(x[:, self.Hp] - x_ref[:, self.Hp], self.Qf)  # 终端代价

        constraints += [(x[:, 0] == X_0)]  # 初始状态约束
        constraints += [(cvxpy.abs(u[0, :]) <= MAX_VELOCITY)]  # 车速区间约束
        constraints += [(cvxpy.abs(u[1, :]) <= MAX_STEER_ANGLE)]  # 方向盘转角约束

        # 创建一个凸优化问题，输入：问题的目标（代价最小），约束
        problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        problem.solve(solver=cvxpy.ECOS, verbose=False)  # 使用 ECOS 求解器进行求解，不输出过程信息

        # 如果求解得到最优解或者近似最优解
        if problem.status == cvxpy.OPTIMAL or problem.status == cvxpy.OPTIMAL_INACCURATE:
            optimal_x = self.matrix_to_ndarray(x.value[0, :])
            optimal_y = self.matrix_to_ndarray(x.value[1, :])
            optimal_yaw = self.matrix_to_ndarray(x.value[2, :])
            optimal_v = self.matrix_to_ndarray(u.value[0, :])
            optimal_delta = self.matrix_to_ndarray(u.value[1, :])
        else:
            print("ERROR: Cannot solve MPC...")
            optimal_x, optimal_y, optimal_yaw, optimal_v, optimal_delta = None, None, None, None, None

        return optimal_x, optimal_y, optimal_yaw, optimal_v, optimal_delta
