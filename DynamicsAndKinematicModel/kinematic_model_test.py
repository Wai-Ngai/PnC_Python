import kinematic_model
import numpy as np


def main():
    model = kinematic_model.LateralErrorModel(m=1000.0, Vx=20.0, C_alpha_f=10000.0, C_alpha_r=15000.0, lf=1.5, lr=1.0,
                                              Iz=3000., g=9.8)
    initial_state = np.array([0.0, 0.0, 0.0,0.0])
    delta_input = 0.1
    psi_des_input = 0.0
    phi_input = 0.1
    dt = 0.1

    state_derivative = model.compute_state_derivative(initial_state, delta_input, psi_des_input, phi_input)
    A_bar, B_bar = model.discrete_state_space(dt)

    print("state derivative", state_derivative)
    print("A_bar", A_bar)
    print("B_bar", B_bar)


if __name__ == '__main__':
    main()
