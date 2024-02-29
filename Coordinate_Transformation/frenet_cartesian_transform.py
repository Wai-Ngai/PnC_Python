import numpy as np
import math


def Frenet2Cartesian(s, s_dot, s_dot2, l, l_dot, l_dot2, dl, ddl, x_r, y_r, theta_r, k_r, dk_r):
    """
          Frenet坐标系转换成Cartesian坐标系
    Args:
        s: 纵向位移，即Frenet纵坐标
        s_dot: Frenet纵向速度
        s_dot2: Frenet纵向加速度
        l: 横向位移，即Frenet横坐标
        l_dot: Frenet横向速度
        l_dot2: Frenet横向加速度
        dl: 横向位移对弧长s的一阶导数
        ddl: 横向位移对弧长s的二阶导数
        x_r: 投影点的x坐标
        y_r: 投影点的y坐标
        theta_r: 投影点的heading
        k_r: 投影点处的曲率
        dk_r: 曲率对弧长s的一阶导数

    Returns:

    """

    x = x_r - l * math.sin(theta_r)
    y = y_r + l * math.cos(theta_r)
    theta_x = math.atan2(dl / (1 - k_r * l)) + theta_r

    delta_theta = theta_x - theta_r

    v_x = math.sqrt((s_dot * (1 - k_r * l)) ** 2 + (s_dot * dl) ** 2)

    k_x = ((ddl + (dk_r * l + k_r * dl) * math.tan(delta_theta) * math.cos(delta_theta) ** 2 / (
            1 - k_r * l)) + k_r) * math.cos(delta_theta) / (1 - k_r * l)
    a_x = s_dot2 * (1 - k_r * l) / math.cos(delta_theta) + s_dot ** 2 / math.cos(delta_theta) * (
            dl * (k_x * (1 - k_r * l) / math.cos(delta_theta) - k_r) - (dk_r * l + k_r * dl))

    return x, y, theta_x, v_x, a_x, k_x


def Cartesian2Frenet(x, y, theta_x, v_x, a_x, k_x, s_r, x_r, y_r, theta_r, k_r, dk_r):
    """

    Args:
        x: Cartesian坐标系下车辆的横向位置
        y: Cartesian坐标系下车辆的纵向位置
        theta_x: 方位角，即全局坐标系下车辆的朝向
        v_x: Cartesian坐标系下的线速度大小
        a_x: Cartesian坐标系下的加速度
        k_x: 车辆行驶轨迹的曲率
        s_r: 投影点的弧长
        x_r: 投影点P在Cartesian坐标系下的x坐标
        y_r: 投影点P在Cartesian坐标系下的y坐标
        theta_r: 投影点P在Cartesian坐标系下的航向角
        k_r: 投影点P的曲率
        dk_r: 曲率对弧长s的一阶导数

    Returns:

    """
    delta_theta = theta_x - theta_r

    s = s_r
    l = np.sign((y - y_r) * math.cos(theta_r) - (x_r - x_r) * math.sin(theta_r)) * math.sqrt(
        (x - x_r) ** 2 + (y - y_r) ** 2)

    one_kr_d = 1 - k_r * l

    l_dot = v_x * math.sin(delta_theta)
    l_dot2 = a_x * math.sin(delta_theta)
    dl = one_kr_d * math.tan(delta_theta)
    ddl = -(dk_r * l + k_r * dl) * math.tan(delta_theta) + one_kr_d / (math.cos(delta_theta) ** 2) * (
            k_x * one_kr_d / math.cos(delta_theta) - k_r)
    s_dot = v_x * math.cos(delta_theta) / one_kr_d
    s_dot2 = (a_x * math.cos(delta_theta) - s_dot ** 2 *
              (dl * (k_x * one_kr_d / math.cos(delta_theta) - k_r) - (dk_r * l + k_r * dl))) / one_kr_d

    return s, s_dot, s_dot2, l, l_dot, l_dot2, dl, ddl
