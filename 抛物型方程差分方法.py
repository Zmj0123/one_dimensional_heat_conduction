import numpy as np
import pandas as pd
import datetime

start_time = datetime.datetime.now()

np.set_printoptions(suppress=True)
#suppress=True表示小数不需要以科学计数法来表示


def left_boundary(t):  # 左边值
    return np.exp(t)  #e的指数


def right_boundary(t):  # 右边值
    return np.exp(t + 1)


def initial_T(x_max, t_max, delta_x, delta_t, m, n):  # 给温度T初始化，左右底边赋真实值，中间点赋值为0
    T = np.zeros((n + 1, m + 1))  #整个网格上有（n+1）*（m+1）个点，初步全赋值为0
    for i in range(m + 1):  #初值
        T[0, i] = np.exp(i * delta_x)  #底边值

    for i in range(1, n + 1):  # 注意不包括T[0,0]与T[0,-1]，这两项已在上一步完成赋值
        T[i, 0] = left_boundary(i * delta_t)  # 左边值
        T[i, -1] = right_boundary(i * delta_t)  # 右边值
    return T  #返回初始化后的矩阵


# 一、向前Euler格式（运用递推关系）
def one_dimensional_heat_conduction1(T, m, n, r):
    # 可以发现当r>=0.5时就发散了
    for k in range(1, n + 1):  # 时间层
        for i in range(1, m):  # 空间层
            T[k, i] = (1 - 2 * r) * T[k - 1, i] + r * (T[k - 1, i - 1] + T[k - 1, i + 1])  # 此题中f(x,y)为0

    return T.round(6)  # 返回T，T中数据四舍五入保留6位小数


# 二、向后Euler格式（运用矩阵乘逆求方程组）
def one_dimensional_heat_conduction2(T, m, n, r):
    A = np.eye(m - 1, k=0) * (1 + 2 * r) + np.eye(m - 1, k=1) * (-r) + np.eye(m - 1, k=-1) * (-r)
    # np.eye(m - 1, k=0)构造m-1阶对角线元素为1的方阵；np.eye（3，4）
    # 在np.eye中，k为0，表示主对角线；k为正值表示上对角线；k为负值表示下对角线

    F = np.zeros(m - 1)  # m-1个元素，索引0~(m-2)
    # np.zeros用法同np.ones
    for k in range(1, n + 1):  # 时间层range(1, n + 1)
        F[0] = T[k - 1, 1] + r * T[k, 0]
        F[-1] = T[k - 1, m - 1] + r * T[k, m]
        for i in range(1, m - 2):  # 空间层
            F[i] = T[k - 1, i + 1]  # 给F赋值
        for i in range(1, m - 1):
            T[k, 1:-1] = np.linalg.inv(A) @ F  # 左乘A逆
            # T[k, 1:-1]是矩阵的索引切片，表示第k行，第1列到倒数第2列
            # T[a:b,c:d],T[a,b],T[a,c:d]；索引切片算头不算尾，负1即表示倒数第1个
    return T.round(6)


# 三、向后Euler格式（追赶法）
def one_dimensional_heat_conduction3(T, m, n, r):
    a = np.ones(m - 1) * (-r)
    a[0] = 0
    # np.ones(m - 1)构造元素全为1的行向量；np.ones((3,4))构造3行4列元素全为1的矩阵
    b = np.ones(m - 1) * (1 + 2 * r)
    c = np.ones(m - 1) * (-r)
    c[-1] = 0
    # c[-1]逆序索引，为逆序第一个元素

    F = np.zeros(m - 1)  # m-1个元素，索引0~(m-2)
    for k in range(1, n + 1):  # 时间层range(1, n + 1)
        F[0] = T[k - 1, 1] + r * T[k, 0]
        F[-1] = T[k - 1, m - 1] + r * T[k, m]
        for i in range(1, m - 2):  # 空间层
            F[i] = T[k - 1, i + 1]  # 给F赋值

        y = np.zeros(m - 1)  # g
        beta = np.zeros(m - 1)  # w
        x = np.zeros(m - 1)  # u
        y[0] = F[0] / b[0]  # g0
        beta[0] = c[0] / b[0]  # w0
        for i in range(1, m - 1):  # 追
            beta[i] = c[i] / (b[i] - a[i] * beta[i - 1])
            y[i] = (F[i] - a[i] * y[i - 1]) / (b[i] - a[i] * beta[i - 1])
        x[-1] = y[-1]  # un=gn
        for i in range(m - 3, -1, -1):  # 赶
            x[i] = y[i] - beta[i] * x[i + 1]
        T[k, 1:-1] = x
    return T.round(6)


# 四、Crank-Nicolson（乘逆矩阵法）
def one_dimensional_heat_conduction4(T, m, n, r):
    A = np.eye(m - 1, k=0) * (1 + r) + np.eye(m - 1, k=1) * (-r * 0.5) + np.eye(m - 1, k=-1) * (-r * 0.5)
    C = np.eye(m - 1, k=0) * (1 - r) + np.eye(m - 1, k=1) * (0.5 * r) + np.eye(m - 1, k=-1) * (0.5 * r)

    for k in range(1, n + 1):  # 时间层
        F = np.zeros(m - 1)  # m-1个元素，索引0~(m-2)
        F[0] = r / 2 * (T[k - 1, 0] + T[k, 0])
        F[-1] = r / 2 * (T[k - 1, m] + T[k, m])
        F = C @ T[k - 1, 1:m] + F
        T[k, 1:-1] = np.linalg.inv(A) @ F
    return T.round(6)


# 五、Crank-Nicolson（追赶法）
def one_dimensional_heat_conduction5(T, m, n, r):
    C = np.eye(m - 1, k=0) * (1 - r) + np.eye(m - 1, k=1) * (0.5 * r) + np.eye(m - 1, k=-1) * (0.5 * r)

    a = np.ones(m - 1) * (-0.5 * r)
    a[0] = 0
    b = np.ones(m - 1) * (1 + r)
    c = np.ones(m - 1) * (-0.5 * r)
    c[-1] = 0

    for k in range(1, n + 1):  # 时间层
        F = np.zeros(m - 1)  # m-1个元素，索引0~(m-2)
        F[0] = r * 0.5 * (T[k - 1, 0] + T[k, 0])
        F[-1] = r * 0.5 * (T[k - 1, m] + T[k, m])
        F = C @ T[k - 1, 1:m] + F

        y = np.zeros(m - 1)  # g
        beta = np.zeros(m - 1)  # w
        x = np.zeros(m - 1)  # u
        y[0] = F[0] / b[0]  # g0
        beta[0] = c[0] / b[0]  # w0
        for i in range(1, m - 1):
            beta[i] = c[i] / (b[i] - a[i] * beta[i - 1])
            y[i] = (F[i] - a[i] * y[i - 1]) / (b[i] - a[i] * beta[i - 1])
        x[-1] = y[-1]  # un=gn
        for i in range(m - 3, -1, -1):  # 赶
            x[i] = y[i] - beta[i] * x[i + 1]
        T[k, 1:-1] = x
    return T.round(6)


def exact_solution(T, m, n, r, delta_x, delta_t):  # 偏微分方程精确解
    for i in range(n + 1):
        for j in range(m + 1):
            T[i, j] = np.exp(i * delta_t + j * delta_x)
    return T.round(6)


a = 1  # 热传导系数
x_max = 1
t_max = 1
delta_x = 0.1  # 空间步长
delta_t = 0.1  # 时间步长
m = int((x_max / delta_x).__round__(4))  # 长度等分成m份
n = int((t_max / delta_t).__round__(4))  # 时间等分成n份
t_grid = np.arange(0, t_max + delta_t, delta_t)  # 时间网格
x_grid = np.arange(0, x_max + delta_x, delta_x)  # 位置网格
r = (a * delta_t / (delta_x ** 2)).__round__(6)  # 网格比
T = initial_T(x_max, t_max, delta_x, delta_t, m, n)
print('长度等分成{}份'.format(m))
print('时间等分成{}份'.format(n))
print('网格比=', r)

p = pd.ExcelWriter('3.有限差分法-一维热传导-（h=0.1;t=0.1）.xlsx')

T1 = one_dimensional_heat_conduction1(T, m, n, r)
T1 = pd.DataFrame(T1, columns=x_grid, index=t_grid)  # colums是列号，index是行号
T1.to_excel(p, '向前Euler格式')

T2 = one_dimensional_heat_conduction2(T, m, n, r)
T2 = pd.DataFrame(T2, columns=x_grid, index=t_grid)  # colums是列号，index是行号
T2.to_excel(p, '向后Euler格式（乘逆矩阵法）')

T3 = one_dimensional_heat_conduction3(T, m, n, r)
T3 = pd.DataFrame(T3, columns=x_grid, index=t_grid)  # colums是列号，index是行号
T3.to_excel(p, '向后Euler格式（追赶法）')

T4 = one_dimensional_heat_conduction4(T, m, n, r)
T4 = pd.DataFrame(T4, columns=x_grid, index=t_grid)  # colums是列号，index是行号
T4.to_excel(p, 'Crank-Nicolson格式（乘逆矩阵法）')

T5 = one_dimensional_heat_conduction5(T, m, n, r)
T5 = pd.DataFrame(T5, columns=x_grid, index=t_grid)  # colums是列号，index是行号
T5.to_excel(p, 'Crank-Nicolson格式（追赶法）')

T6 = exact_solution(T, m, n, r, delta_x, delta_t)
T6 = pd.DataFrame(T6, columns=x_grid, index=t_grid)  # colums是列号，index是行号
T6.to_excel(p, '偏微分方程精确解')

p.close()

end_time = datetime.datetime.now()
print('运行时间为', (end_time - start_time))
