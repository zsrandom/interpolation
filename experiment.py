import numpy as np
import time
# from memory_profiler import profile
from interpolation_new import newton_segment_interpolation, lagrange_segment_interpolation
# 插值算法实现
# @profile
def lagrange_interpolation(x_points, y_points, x):
    """
    拉格朗日插值法
    :param x_points: 插值点的 x 坐标列表
    :param y_points: 插值点的 y 坐标列表
    :param x: 需要计算插值的 x 值
    :return: 插值结果 L(x)
    """
    n = len(x_points) - 1  # 插值多项式的次数
    L_x = 0  # 初始化插值结果

    # 遍历每个基函数
    for i in range(n + 1):
        l_i_x = 1  # 初始化基函数 l_i(x)

        # 计算基函数 l_i(x)
        for j in range(n + 1):
            if i != j:
                l_i_x *= (x - x_points[j]) / (x_points[i] - x_points[j])

        # 累加基函数与 y_i 的乘积
        L_x += y_points[i] * l_i_x

    return L_x


# @profile
def newton_interpolation(x_points, y_points, x):
    """
    牛顿插值法
    :param x_points: 插值点的 x 坐标列表
    :param y_points: 插值点的 y 坐标列表
    :param x: 需要计算插值的 x 值
    :N_x: 插值结果 N(x)
    """
    n = len(x_points)  # 插值点的数量
    fdd = [[0 for _ in range(n)] for _ in range(n)]  # 初始化差商表

    # 计算零阶差商（即 y 值本身）
    for i in range(n):
        fdd[i][0] = y_points[i]

    # 计算高阶差商
    for j in range(1, n):
        for i in range(n - j):
            fdd[i][j] = (fdd[i + 1][j - 1] - fdd[i][j - 1]) / (x_points[i + j] - x_points[i])

    # 构建牛顿插值多项式并计算结果
    N_x = fdd[0][0]
    product = 1
    for j in range(1, n):
        product *= (x - x_points[j - 1])
        N_x += fdd[0][j] * product

    return N_x


# 精度评估
def evaluate_precision(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)  # 均方误差
    mae = np.max(np.abs(y_true - y_pred))  # 最大绝对误差
    return mse, mae


# 计算时间测量
def measure_interpolation_time(x_points, y_points, interpolation_func, test_x):
    """
    测量插值函数的运行时间
    :param x_points: 插值点的 x 坐标列表
    :param y_points: 插值点的 y 坐标列表
    :param interpolation_func: 插值函数
    :param test_x: 测试点的 x 值列表
    :return: 运行时间
    """
    start_time = time.time()
    for x in test_x:
        interpolation_func(x_points, y_points, x)
    end_time = time.time()
    return end_time - start_time


# 数据生成
def generate_data1(func=lambda x: x, interval=(-1, 1), n=10, noise_std=0.1):
    x_points = np.linspace(interval[0], interval[1], n)  # 均匀采样n个点
    y_points = func(x_points) + np.random.normal(0, noise_std, n)  # 添加高斯噪声
    return x_points, y_points


def generate_data2(func=lambda x: x, interval=(-1, 1), n=10, noise_std=0.1):
    x_points = np.random.uniform(interval[0], interval[1], n)
    x_points = np.sort(x_points)  # 排序
    y_points = func(x_points) + np.random.normal(0, noise_std, n)
    return x_points, y_points


def experiment(func, interval, n, noise_std, generate_data, m):
    x_points, y_points = generate_data(func, interval, n, noise_std)
    test_x = np.linspace(interval[0], interval[1], m)
    test_y = func(test_x)

    lagrange_y = np.array([lagrange_interpolation(x_points, y_points, x) for x in test_x])
    lagrange_time = measure_interpolation_time(x_points, y_points, lagrange_interpolation, test_x)
    lagrange_mse, lagrange_mae = evaluate_precision(test_y, lagrange_y)

    newton_y = np.array([newton_interpolation(x_points, y_points, x) for x in test_x])
    newton_time = measure_interpolation_time(x_points, y_points, newton_interpolation, test_x)
    newton_mse, newton_mae = evaluate_precision(test_y, newton_y)

    print(x_points)
    print(f"拉格朗日插值 - MSE: {lagrange_mse}, MAE: {lagrange_mae}, 时间: {round(lagrange_time, 5)}")
    print(f"牛顿插值 - MSE: {newton_mse}, MAE: {newton_mae}, 时间: {round(newton_time, 5)}")




def experiment0(func, interval, n, noise_std, generate_data, m):
    x_points, y_points = generate_data(func, interval, n, noise_std)
    test_x = np.linspace(interval[0], interval[1], m)
    test_y = func(test_x)

    lagrange_y = np.array([lagrange_segment_interpolation(x_points, y_points, x) for x in test_x])
    lagrange_time = measure_interpolation_time(x_points, y_points, lagrange_segment_interpolation, test_x)
    lagrange_mse, lagrange_mae = evaluate_precision(test_y, lagrange_y)

    newton_y = np.array([newton_segment_interpolation(x_points, y_points, x) for x in test_x])
    newton_time = measure_interpolation_time(x_points, y_points, newton_segment_interpolation, test_x)
    newton_mse, newton_mae = evaluate_precision(test_y, newton_y)

    print(x_points)
    print(f"拉格朗日插值 - MSE: {lagrange_mse}, MAE: {lagrange_mae}, 时间: {round(lagrange_time, 5)}")
    print(f"牛顿插值 - MSE: {newton_mse}, MAE: {newton_mae}, 时间: {round(newton_time, 5)}")



def experiment1(func, interval, n, noise_std, generate_data, m):
    x_points, y_points = generate_data(func, interval, n, noise_std)
    test_x = np.linspace(interval[0], interval[1], m)
    test_y = func(test_x)
    lagrange_time = measure_interpolation_time(x_points, y_points, lagrange_interpolation, test_x)
    newton_time = measure_interpolation_time(x_points, y_points, newton_interpolation, test_x)
    print(f"拉格朗日插值  时间: {round(lagrange_time, 5)}")
    print(f"牛顿插值 时间: {round(newton_time, 5)}")


def experiment2(func, interval, n, noise_std, generate_data, m):
    x_points, y_points = generate_data(func, interval, n, noise_std)
    test_x = np.linspace(interval[0], interval[1], m)
    test_y = func(test_x)
    lagrange_time = measure_interpolation_time(x_points, y_points, lagrange_segment_interpolation, test_x)
    newton_time = measure_interpolation_time(x_points, y_points, newton_segment_interpolation, test_x)
    print(f"拉格朗日插值  时间: {round(lagrange_time, 5)}")
    print(f"牛顿插值 时间: {round(newton_time, 5)}")


if __name__ == "__main__":
    func = np.sin
    # print("=============================Experiment_1: 低噪声均匀数据测试=======================================")
    # experiment0(func, [-10, 10], 10, 0.01, generate_data1, 100)

    # print("=============================Experiment_2: 高噪声非均匀数据测试=====================================")
    # b = 10
    # i = 1
    # while b:
    #     print("========================================第", i, "次循环============================================")
    #     experiment0(func, [-10, 10], 10, 1.0, generate_data2, 200)
    #     b -= 1
    #     i += 1
    # print("=============================Experiment_2: 高噪声非均匀数据测试=====================================")
    # b = 5
    # while b:
    #     print("===============================================================================================")
    #     experiment0(func, [-10, 10], 5, 1.0, generate_data2, 10)
    #     b -= 1

    func1 = lambda x: x**2
    print("==============================Experiment_3: 大规模数据集测试=======================================")
    print("1000个插值点：")
    experiment2(func1, [-1000, 1000], 100, 0, generate_data1, 1000)
    print("2000个插值点：")
    experiment2(func1, [-1000, 1000], 100, 0, generate_data1, 2000)
    print("5000个插值点：")
    experiment2(func1, [-1000, 1000], 100, 0, generate_data1, 5000)
    print("10000个插值点：")
    experiment2(func1, [-1000, 1000], 100, 0, generate_data1, 10000)
    print("==============================Experiment_3: 大规模数据集测试=======================================")
    print("1000个插值点：")
    experiment1(func1, [-1000, 1000], 100, 0, generate_data1, 1000)
    print("2000个插值点：")
    experiment1(func1, [-1000, 1000], 100, 0, generate_data1, 2000)
    print("5000个插值点：")
    experiment1(func1, [-1000, 1000], 100, 0, generate_data1, 5000)
    print("10000个插值点：")
    experiment1(func1, [-1000, 1000], 100, 0, generate_data1, 10000)
