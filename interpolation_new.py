import numpy as np


def newton_divided_diff(x, y):
    n = len(x)
    coef = y.copy()
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])
    return coef


def newton_evaluate(x, coef, x_interp):
    n = len(coef)
    result = coef[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x_interp - x[i]) + coef[i]
    return result


def newton_segment_interpolation(x_points, y_points, x_query, n_segments=8, poly_degree=2):
    """
    分段牛顿插值算法
    :param x_points: 已知数据点的x坐标（需升序）
    :param y_points: 已知数据点的y坐标
    :param x_query: 待插值的x值（可以是标量或数组）
    :param n_segments: 区间分段数
    :param poly_degree: 每段插值的多项式次数
    :return: 插值结果 y_interp（列表）
    """
    # 排序
    sorted_idx = np.argsort(x_points)
    x_sorted = np.array(x_points)[sorted_idx]
    y_sorted = np.array(y_points)[sorted_idx]

    # 划分分段区间
    x_min, x_max = x_sorted[0], x_sorted[-1]
    segment_bounds = np.linspace(x_min, x_max, n_segments + 1)

    # 插值结果容器
    y_interp = []

    for x in np.atleast_1d(x_query):
        # 确定属于哪个分段
        seg_idx = np.searchsorted(segment_bounds, x) - 1
        seg_idx = np.clip(seg_idx, 0, n_segments - 1)

        # 获取当前段内的点
        mask = (x_sorted >= segment_bounds[seg_idx]) & (x_sorted <= segment_bounds[seg_idx + 1])
        x_seg = x_sorted[mask]
        y_seg = y_sorted[mask]

        # 确保至少选取 poly_degree + 1 个点
        if len(x_seg) < poly_degree + 1:
            # 若不够，从全局中就近补充点
            distances = np.abs(x_sorted - x)
            nearest_idx = np.argsort(distances)[:poly_degree + 1]
            x_seg = x_sorted[nearest_idx]
            y_seg = y_sorted[nearest_idx]

        # 构建差商系数
        coef = newton_divided_diff(x_seg, y_seg)
        y = newton_evaluate(x_seg, coef, x)
        y_interp.append(y)

    return y_interp



def lagrange_segment_interpolation(x_points, y_points, x_query, n_segments=8, poly_degree=2):
    """
    分段拉格朗日插值算法
    :param x_points: 已知数据点的x坐标列表（需按升序排列）
    :param y_points: 已知数据点的y坐标列表
    :param x_query: 待插值的x值（标量或数组）
    :param n_segments: 分段数量（默认3段）
    :param poly_degree: 每段多项式的次数（默认2次）
    :return: 插值结果y值
    """
    # 将数据点按x排序
    sorted_indices = np.argsort(x_points)
    x_sorted = np.array(x_points)[sorted_indices]
    y_sorted = np.array(y_points)[sorted_indices]

    # 划分区间边界
    x_min, x_max = x_sorted[0], x_sorted[-1]
    segment_bounds = np.linspace(x_min, x_max, n_segments + 1)

    # 对每个分段进行插值
    y_interp = []
    for x in np.atleast_1d(x_query):
        # 确定x所属的分段
        segment_idx = np.searchsorted(segment_bounds, x) - 1
        segment_idx = np.clip(segment_idx, 0, n_segments - 1)

        # 提取分段内的数据点
        mask = (x_sorted >= segment_bounds[segment_idx]) & (x_sorted <= segment_bounds[segment_idx + 1])
        x_segment = x_sorted[mask]
        y_segment = y_sorted[mask]

        # 如果分段内点数不足，自动降低多项式次数
        valid_degree = min(poly_degree, len(x_segment) - 1)

        # 拉格朗日插值计算
        y = 0.0
        for i in range(len(x_segment)):
            term = y_segment[i]
            for j in range(len(x_segment)):
                if j != i:
                    term *= (x - x_segment[j]) / (x_segment[i] - x_segment[j])
            y += term
        y_interp.append(y)

    return y_interp

