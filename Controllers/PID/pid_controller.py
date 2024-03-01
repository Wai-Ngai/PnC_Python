class PIDController1(object):
    """
        PID控制器，位置式
    """

    def __init__(self, kp, ki, kd, target, upper=1.0, lower=-1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.upper_bound = upper
        self.lower_bound = lower

        self.err_last = 0
        self.err_sum = 0
        self.previous_output = 0
        self.first_hit = True

    def set_k(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_bounds(self, upper, lower):
        self.upper = upper
        self.lower = lower

    def reset(self):
        self.err_last = 0
        self.err_sum = 0

    def calculate(self, error, dt):
        if dt < 0:
            print("dt must be greater than zero")
            return self.previous_output

        # 差分计算
        if self.first_hit:
            self.first_hit = False
            diff = 0
        else:
            diff = (error - self.err_last) / dt

        # 积分及积分保持
        integral = self.err_sum + self.ki * error * dt
        if integral > self.upper_bound:
            integral = self.upper_bound
        elif integral < self.lower_bound:
            integral = self.lower_bound

        # PID计算
        output = self.kp * error + integral + self.kd * diff
        self.previous_output = output
        self.err_sum = integral

        return output


class PIDController2(object):
    """
        PID控制器，增量式
    """

    def __init__(self, kp, ki, kd, upper=1.0, lower=-1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.upper_bound = upper
        self.lower_bound = lower

        self.error = 0
        self.error_last = 0
        self.error_last_prev = 0
        self.increase = 0
        self.output = 0

    def set_k(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_bounds(self, upper, lower):
        self.upper_bound = upper
        self.lower_bound = lower

    def reset(self):
        self.error = 0
        self.error_last = 0
        self.error_last_prev = 0

    def calculate(self, error):
        self.increase = self.kd * (error - self.error_last) + self.ki * error + self.kd * (
                error - 2 * self.error_last + self.error_last_prev)

        self.error_last_prev = self.error_last
        self.error_last = error
        self.output += self.increase

        if self.output > self.upper_bound:
            self.output = self.upper_bound
        elif self.output < self.lower_bound:
            self.output = -self.lower_bound

        return self.output
