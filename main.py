import numpy as np
import LinearMPCFactor as lMf

if __name__ == '__main__':
    A = np.array([[2, 1], [0, 2]])  # state space equation A 状态空间方程中的A
    B = np.array([[1, 0], [0, 1]])  # state space equation B 状态空间方程中的B
    Q = np.array([[1, 0], [0, 3]])  # cost function Q, which determines the convergence rate of the state 代价函数中的Q，决定了状态的收敛速度
    R = np.array([[1, 0], [0, 1]])  # cost function R, which determines the convergence rate of the input 代价函数中的R，决定了输入的收敛速度

    A_x = np.array([[1, 0], [-1, 0]])  # state constraints A_x @ x_k <= b_x 状态约束 A_x @ x_k <= b_x
    b_x = np.array([5, 5])

    A_u = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])  # input constraints A_u @ u_k <= b_u 输入约束 A_u @ u_k <= b_u
    b_u = np.array([1, 1, 1, 1])

    N = 5  # prediction horizon 预测区间

    mpc = lMf.LinearMPCFactor(A, B, Q, R, N, A_x, b_x, A_u, b_u)  # print the cpp code for initializing a MPC class 打印出初始化MPC类的cpp代码

    # print(mpc.QN)
    # print(mpc.F)
    # print(mpc.G)
    # print(mpc.c)
    # print(mpc.FN)
    # print(mpc.cN)
    # print(mpc.L_phi)

    # mpc.decPlace = 4  # This is the number of decimal places reserved for matrix data, which defaults to 6 decimal places 这是矩阵数据保留的小数位数，默认保留小数点后6位
    e_V = 0.001  # tolerance of the error between optimal cost and real cost 实际代价函数的值与最优之间的最大误差
    e_g = 0.001  # tolerance of the violation of constraints 最大的违反约束的程度
    max_iter = 1000  # maximum steps of the solver 最大迭代步数

    mpc.PrintCppCode(e_V, e_g, max_iter)
