# coding=utf-8
"""
单变量数据可视化，用于初步的探索性数据分析
author:K.Lz <565150134@qq.com>
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# ===============================================================
# 数值型                                                       ##
# 1.直方图                                                     ##
#                                                              ##
# 类别型                                                       ##
# 1.条形图                                                     ##
# ===============================================================

# 直方图_1


def compare_hist_diagram(data, col_name, bin1, bin2):
    """
    用来比较直方图设置不同bin值时的影响(*^▽^*)
    used to compare different bin values
    :param data:传入的dataframe
    :param col_name:需要作图的列
    :param bin1:比较的bin值1
    :param bin2:比较的bin值2
    :return:None ,plotting
    """
    plt.figure(figsize=[10, 5])  # larger figure size for subplots

    left_val = data[col_name].min()
    right_val = data[col_name].max()

    # first_bin
    plt.subplot(1, 2, 1)  # 1 row, 2 cols, subplot 1
    bin_edges = np.arange(left_val, right_val + bin1, bin1)
    plt.hist(data=data, x=col_name, bins=bin_edges)
    plt.xlabel('histogram_1: ' + col_name + ' bin = ' + str(bin1))

    # second_bin
    plt.subplot(1, 2, 2)  # 1 row, 2 cols, subplot 2
    bin_edges = np.arange(left_val, right_val + bin2, bin2)
    plt.hist(data=data, x=col_name, bins=bin_edges)
    plt.xlabel('histogram_2: ' + col_name + ' bin = ' + str(bin2))

    plt.show()


def compare_hist_diagram_plus(data, col_name, bin_list):
    """
    用来比较直方图设置不同bin值时的影响(*^▽^*)，plus版本，可以设置一系列的bin值
    used to compare different bin values
    :param data:传入的dataframe
    :param col_name:需要作图的列
    :param bin_1ist:比较一系列的bin值
    :return:None ,plotting
    """
    left_val = data[col_name].min()
    right_val = data[col_name].max()

    fig_num = len(bin_list)
    fig_col = 3   # 一行最多画3张图
    fig_row = math.ceil(fig_num / fig_col)
    plt.figure(figsize=[5 * fig_col, 5 * fig_row])  # larger figure size for subplots

    for i in range(fig_num):
        plt.subplot(fig_row, fig_col, i+1)
        bin_edges = np.arange(left_val, right_val + bin_list[i], bin_list[i])
        plt.hist(data=data, x=col_name, bins=bin_edges)
        plt.xlabel(f'histogram_{i}: ' + col_name + ' bin = ' + str(bin_list[i]))
    plt.show()


# 条形图_1
def super_count_plot(data, col_name, color_index=0, order_list=None, if_prop = False, prop_num = 0.05):
    """
    用来绘制类别变量的条形图
    :param data:待画图的dataframe
    :param col_name:给定需要画图的列名，给定数值
    :param color_index:指定作图颜色在调色板上的序列号，如果为空则默认为0(蓝色)
    :param order_list:指定类别顺序，如果为空则按照统计后从高到低的顺序作图
    :param if_prop:bool,指定是否展示相对频率。False：绝对频率；True：相对频率。
    :param prop_num:float,指定相对频率画图时，纵坐标刻度线标注份数或者区间长度，小于1为区间长度，大于等于1为份数。
    :return:one ,plotting
    """
    base_color = sns.color_palette()[color_index]
    if if_prop:
        n_points = data.shape[0]
        # 最多类别出现的次数
        max_count = data[col_name].value_counts().max()
        max_prop = max_count / n_points  # 最大比例

        # 生成比例
        if prop_num < 1:
            unit_prop = prop_num
        else:
            unit_prop = max_prop / prop_num
        tick_props = np.arange(0, max_prop, unit_prop)
        tick_names = ['{:0.2f}'.format(v) for v in tick_props]
    if order_list:
        try:
            # 根据pandas版本，pandas 为 0.20.3 或更低版本这里可能会报错
            ordered_cat = pd.api.types.CategoricalDtype(ordered=True, categories=order_list)
            data[col_name] = data[col_name].astype(ordered_cat)
        except:
            data[col_name] = data[col_name].astype('category', ordered=True, categories=order_list)
        if not if_prop:
            sns.countplot(data=data, x=col_name, color=base_color)
        else:
            sns.countplot(data=data, x=col_name, color=base_color)
            plt.yticks(tick_props * n_points, tick_names)
            plt.ylabel('proportion')
    else:
        cat_order = data[col_name].value_counts().index
        if not if_prop:
            sns.countplot(data=data, x=col_name, color=base_color, order=cat_order)
        else:
            sns.countplot(data=data, x=col_name, color=base_color, order=cat_order)
            plt.yticks(tick_props * n_points, tick_names)
            plt.ylabel('proportion')


if __name__ == '__main__':
    # 鸢尾花数据集
    from sklearn.datasets import load_iris

    df = pd.DataFrame(load_iris()['data'], columns=load_iris().feature_names)
    compare_hist_diagram(df, 'petal length (cm)', 1, 0.5)

    compare_hist_diagram_plus(df, 'petal length (cm)', [1, 0.5,0.1,0.2])

    df1 = pd.read_csv(r"D:\金融数据\账户静态信息.csv")
    # base_color = sns.color_palette()[0]
    # sns.countplot(data=df, x='xb', color=1)
    super_count_plot(df1, col_name='xb', color_index=2, order_list=[1, 0])
    super_count_plot(df1, col_name='xb', color_index=2)
    super_count_plot(df1, col_name='xb', color_index=2, order_list=[1, 0],if_prop=True)