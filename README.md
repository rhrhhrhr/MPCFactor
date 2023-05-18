# MPCFactor
This is used for generating the setup code for the MPC controller Arduino Library MPC_ruih
## Usage
To generate the setup code for MPC class in MPC_ruih library, you can use the python package `LinearMPCFactor` like the way in [main.py](https://github.com/rhrhhrhr/MPC_ruih_SolverSetup/blob/main/main.py):
```python
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

    mpc = lMf.LinearMPCFactor(A, B, Q, R, 5, A_x, b_x, A_u, b_u)

    # mpc.decPlace = 4  # This is the number of decimal places reserved for matrix data, which defaults to 6 decimal places 这是矩阵数据保留的小数位数，默认保留小数点后6位

    mpc.PrintCppCode(0.001, 0.001, 1000)
```
**result:**
```cpp
MatDataType_t A_arr[4] = {2, 1, 0, 2};
MatDataType_t B_arr[4] = {1, 0, 0, 1};
MatDataType_t Q_arr[4] = {1, 0, 0, 3};
MatDataType_t R_arr[4] = {1, 0, 0, 1};
MatDataType_t QN_arr[4] = {4.167039, 1.756553, 1.756553, 7.455801};
MatDataType_t F_arr[12] = {1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
MatDataType_t G_arr[12] = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0};
MatDataType_t c_arr[6] = {5, 5, 1, 1, 1, 1};
MatDataType_t FN_arr[8] = {1.583519, 0.878277, -1.583519, -0.878277, 0.086517, 1.788762, -0.086517, -1.788762};
MatDataType_t cN_arr[4] = {1.0, 1.0, 1.0, 1.0};

Matrix A = Matrix(2, 2);
Matrix B = Matrix(2, 2);
Matrix Q = Matrix(2, 2);
Matrix R = Matrix(2, 2);
Matrix QN = Matrix(2, 2);
Matrix F = Matrix(6, 2);
Matrix G = Matrix(6, 2);
Matrix c = Matrix(6, 1);
Matrix FN = Matrix(4, 2);
Matrix cN = Matrix(4, 1);

A = A_arr;
B = B_arr;
Q = Q_arr;
R = R_arr;
QN = QN_arr;
F = F_arr;
G = G_arr;
c = c_arr;
FN = FN_arr;
cN = cN_arr;

MatDataType_t L_phi = 9.90287;

MPC mpc = MPC(L_phi, 0.001, 0.001, 1000, 5, A, B, Q, R, QN, F, G, c, FN, cN);
```
## Library Version
numpy==1.22.4<br>
scipy==1.7.3
