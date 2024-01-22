class PIDController(object):
    """
    PID控制器 位置式
    """

    def __init__(self, Kp, Ki, Kd, upper = 1.0, lower = -1.0):
        self.error_sum = 0
        self.previous_error = 0
        self.previous_output = 0
        self.first_hit = True

        self.upper_bound = upper
        self.lower_bound = lower
        self.set_pid(Kp, Ki, Kd)

    def set_pid(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd

    def reset_pid(self):
        self.err = 0
        self.err_last = 0

    def reset_integral(self):
        self.error_sum =0

    def calculate(self, error, dt):
        if dt <= 0:
            print("dt must be greater than zero")
            return self.previous_output

        # 差分计算
        if self.first_hit:
            self.first_hit = False
            diff = 0
        else:
            diff = (error - self.previous_error) / dt

        # 积分 及 积分保持
        integral = self.error_sum + self.Ki * error * dt
        if integral > self.upper_bound:
            integral = self.upper_bound
        elif integral < self.lower_bound:
            integral = self.lower_bound

        # PID计算
        output = self.Kp * error + integral + self.Kd * diff
        self.previous_output = output
        self.error_sum = integral
        return output
