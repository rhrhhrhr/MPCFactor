import numpy as np
import LinearMPCFactor as lMf

if __name__ == '__main__':
    A = np.array([[2, 1], [0, 2]])  # state space equation A
    B = np.array([[1, 0], [0, 1]])  # state space equation B
    Q = np.array([[1, 0], [0, 3]])
    R = np.array([[1, 0], [0, 1]])

    A_x = np.array([[1, 0], [-1, 0]])
    b_x = np.array([5, 5])

    A_u = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b_u = np.array([1, 1, 1, 1])

    mpc = lMf.LinearMPCFactor(A, B, Q, R, 5, A_x, b_x, A_u, b_u)

    # print(mpc.QN)
    # print(mpc.F)
    # print(mpc.G)
    # print(mpc.c)
    # print(mpc.FN)
    # print(mpc.cN)
    # print(mpc.L_phi)

    # mpc.decPlace = 4  # 这是矩阵数据保留的小数位数，默认保留小数点后6位

    mpc.PrintCppCode(0.001, 0.001, 1000)
