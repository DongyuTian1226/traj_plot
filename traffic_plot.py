'''对数据进行聚合等操作后, 对交通数据进行画图'''
import os
from math import nan
import pandas as pd
import matplotlib.pyplot as plt
from .research_plt import ResearchPlt


def section_v_hills(
    df: pd.DataFrame, window_idx: str, window_size: int, window_step: int,
    car_idx = 'vehicleID', time_idx = 'time(s)', lane_idx = 'laneID', x_idx = 'x(m)',
):
    '''基于空间的速度时间山峦图
    每张图像中: 横轴为路段距离, 纵轴为统计窗口路段内车辆的平均速度,
    这些图像按时间顺序, 纵向排布
    具体地：
    对数据进行滑动窗口统计, 从第0行的数据开始, 每step行进行一次统计, 统计窗口大小为window_size行。
    若开头或者末尾几个数据长度不足window_size, 则按照数据长度进行统计。

    input
    -----
    df: 数据, 包含车辆的基本信息, 如: 车辆ID, 时间, 车道ID, 位置等
    window_idx: 聚合列, 一般为时间或位置
    window_size: 统计窗口大小
    window_step: 统计窗口步长, 每次移动的距离
    '''
    pass


def section_k_hills(
    df: pd.DataFrame, window_idx: str, window_size: int, window_step: int,
    car_idx ='vehicleID', time_idx = 'time(s)', lane_idx = 'laneID', x_idx = 'x(m)',
):
    '''基于空间的密度时间山峦图
    每张图像中: 横轴为统计窗口内车辆的数量, 纵轴为统计窗口路段内交通流密度
    这些图像按时间顺序, 纵向排布.
    具体地：
    对数据进行滑动窗口统计, 从第0行的数据开始, 每step行进行一次统计, 统计窗口大小为window_size行。
    若开头或者末尾几个数据长度不足window_size, 则按照数据长度进行统计。

    input
    -----
    df: 数据, 包含车辆的基本信息, 如: 车辆ID, 时间, 车道ID, 位置等
    window_idx: 聚合列, 一般为时间或位置
    window_size: 统计窗口大小
    window_step: 统计窗口步长, 每次移动的距离
    '''
    pass


def plot_ttc(
    df: pd.DataFrame, output_dir: str, thresholds: list = [1, 2, 3],
    car_idx ='vehicleID', time_idx = 'time(s)', lane_idx = 'laneID', x_idx = 'x(m)',
):
    '''绘制碰撞时间ttc分布的时序图, 以及不同ttc阈值的数量情况
    
    input
    -----
    df: 数据, 包含车辆的基本信息, 如: 车辆ID, 时间, 车道ID, 位置等
    output_dir: 输出文件夹
    thresholds: ttc阈值列表, 单位s
    '''
    pass


def plot_ttt(
    df: pd.DataFrame, window_size: int, output_dir: str,
    car_idx ='vehicleID', time_idx = 'time(s)',
):
    '''绘制车辆总旅行时间的时间序列图, 包括全局累计平均ttt和时间窗口内平均ttt。
    具体地：
    对数据进行滑动窗口统计, 从第0行的数据开始, 每step行进行一次统计, 统计窗口大小为window_size行。
    若开头或者末尾几个数据长度不足window_size, 则按照数据长度进行统计。时间窗口step默认为1
    需要按ID氛围两类车, 一类是主线车辆, 一类是匝道车辆

    input
    -----
    df: 数据, 包含车辆的基本信息, 如: 车辆ID, 时间, 车道ID, 位置等
    output_dir: 输出文件夹
    window_size: 统计窗口大小
    '''
    ResearchPlt(legend_loc='upper left')       # 执行全局科研可视化配置
    # 按车辆ID分组，计算每辆车的总旅行时间
    grouped = df.groupby(car_idx)
    ttt_df = grouped[time_idx].agg(['max', 'min']).reset_index()
    ttt_df['ttt(s)'] = ttt_df['max'] - ttt_df['min']

    # 添加车辆类型分类（主线车辆和匝道车辆）
    ttt_df['vehicle_type'] = ttt_df[car_idx].apply(lambda x: 'mainline' if x < 100000 else 'ramp')

    # 以max作为车辆离开时间, 统计时序窗口数据
    # start_time = df[time_idx].min()     # ASSUMPTION: 开始时间值为0
    end_time = int(df[time_idx].max()) + 1
    window_avg_mainline, window_std_mainline = [], []
    window_avg_ramp, window_std_ramp = [], []
    window_avg_all, window_std_all = [], []
    cumulative_avg_mainline, cumulative_std_mainline = [], []
    cumulative_avg_ramp, cumulative_std_ramp = [], []
    cumulative_avg_all, cumulative_std_all = [], []

    for t in range(window_size, end_time + 1):     # 时间步默认为1
        # 计算时间窗口内TTT数据
        window_data = ttt_df[(ttt_df['max'] >= t - window_size) & (ttt_df['max'] < t)]
        mainline_window_data = ttt_df[(ttt_df['max'] >= t - window_size) & (ttt_df['max'] < t) & (ttt_df['vehicle_type'] == 'mainline')]
        ramp_window_data = ttt_df[(ttt_df['max'] >= t - window_size) & (ttt_df['max'] < t) & (ttt_df['vehicle_type'] == 'ramp')]
        window_avg_all.append(window_data['ttt(s)'].mean() if not window_data.empty else nan)
        window_std_all.append(window_data['ttt(s)'].std() if not window_data.empty else nan)
        window_avg_mainline.append(mainline_window_data['ttt(s)'].mean() if not mainline_window_data.empty else nan)
        window_std_mainline.append(mainline_window_data['ttt(s)'].std() if not mainline_window_data.empty else nan)
        window_avg_ramp.append(ramp_window_data['ttt(s)'].mean() if not ramp_window_data.empty else nan)
        window_std_ramp.append(ramp_window_data['ttt(s)'].std() if not ramp_window_data.empty else nan)
        # 计算累计窗口内TTT数据
        window_cumu_data = ttt_df[ttt_df['max'] < t]
        mainline_cumu_data = ttt_df[(ttt_df['max'] < t) & (ttt_df['vehicle_type'] == 'mainline')]
        ramp_cumu_data = ttt_df[(ttt_df['max'] < t) & (ttt_df['vehicle_type'] == 'ramp')]
        cumulative_avg_all.append(window_cumu_data['ttt(s)'].mean() if not window_cumu_data.empty else nan)
        cumulative_std_all.append(window_cumu_data['ttt(s)'].std() if not window_cumu_data.empty else nan)
        cumulative_avg_mainline.append(mainline_cumu_data['ttt(s)'].mean() if not mainline_cumu_data.empty else nan)
        cumulative_std_mainline.append(mainline_cumu_data['ttt(s)'].std() if not mainline_cumu_data.empty else nan)
        cumulative_avg_ramp.append(ramp_cumu_data['ttt(s)'].mean() if not ramp_cumu_data.empty else nan)
        cumulative_std_ramp.append(ramp_cumu_data['ttt(s)'].std() if not ramp_cumu_data.empty else nan)

    # 保存结果到DataFrame
    result_df = pd.DataFrame({
        'time(s)': range(window_size, end_time + 1),
        'window_avg_all': window_avg_all,
        'window_std_all': window_std_all,
        'window_avg_mainline': window_avg_mainline,
        'window_std_mainline': window_std_mainline,
        'window_avg_ramp': window_avg_ramp,
        'window_std_ramp': window_std_ramp,
        'cumulative_avg_all': cumulative_avg_all,
        'cumulative_std_all': cumulative_std_all,
        'cumulative_avg_mainline': cumulative_avg_mainline,
        'cumulative_std_mainline': cumulative_std_mainline,
        'cumulative_avg_ramp': cumulative_avg_ramp,
        'cumulative_std_ramp': cumulative_std_ramp
    })
    # 填补空元素，继承自上一个非空
    result_df.ffill(inplace=True)
    # 增加一行time为0的数据, 对应数值设置为0(画图更标准)
    result_df = pd.concat([pd.DataFrame([{'time(s)': 0}]), result_df], ignore_index=True)
    result_df.fillna(value=0, inplace=True)
    
    

    # 绘制图形
    plt.figure(figsize=(10, 6))
    # 所有车辆
    plt.plot(result_df['time(s)'], result_df['window_avg_all'], label='Window Avg All', color='red')
    plt.fill_between(result_df['time(s)'],
                        result_df['window_avg_all'] - result_df['window_std_all'],
                        result_df['window_avg_all'] + result_df['window_std_all'],
                        color='red', alpha=0.1)
    plt.plot(result_df['time(s)'], result_df['cumulative_avg_all'], label='Cumulative Avg All', color='red', linestyle='--')

    plt.fill_between(result_df['time(s)'],
                        result_df['cumulative_avg_all'] - result_df['cumulative_std_all'],
                        result_df['cumulative_avg_all'] + result_df['cumulative_std_all'],
                        color='darkred', alpha=0.1)
    # 主线车辆
    plt.plot(result_df['time(s)'], result_df['window_avg_mainline'], label='Window Avg Mainline', color='blue')
    plt.fill_between(result_df['time(s)'],
                        result_df['window_avg_mainline'] - result_df['window_std_mainline'],
                        result_df['window_avg_mainline'] + result_df['window_std_mainline'],
                        color='blue', alpha=0.1)
    plt.plot(result_df['time(s)'], result_df['cumulative_avg_mainline'], label='Cumulative Avg Mainline', color='darkblue', linestyle='--')
    plt.fill_between(result_df['time(s)'],
                        result_df['cumulative_avg_mainline'] - result_df['cumulative_std_mainline'],
                        result_df['cumulative_avg_mainline'] + result_df['cumulative_std_mainline'],
                        color='darkblue', alpha=0.1)
    # 匝道车辆
    plt.plot(result_df['time(s)'], result_df['window_avg_ramp'], label='Window Avg Ramp', color='orange', )
    plt.fill_between(result_df['time(s)'],
                        result_df['window_avg_ramp'] - result_df['window_std_ramp'],
                        result_df['window_avg_ramp'] + result_df['window_std_ramp'],
                        color='orange', alpha=0.1)
    plt.plot(result_df['time(s)'], result_df['cumulative_avg_ramp'], label='Cumulative Avg Ramp', color='darkorange', linestyle='--')
    plt.fill_between(result_df['time(s)'],
                        result_df['cumulative_avg_ramp'] - result_df['cumulative_std_ramp'],
                        result_df['cumulative_avg_ramp'] + result_df['cumulative_std_ramp'],
                        color='darkorange', alpha=0.1)
    # 设置标题和标签
    plt.title('Total Travel Time (TTT) Analysis')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Travel Time (s)')
    plt.xlim(0, end_time)
    plt.legend()
    # 添加图例
    plt.legend()
    # 添加网格
    plt.grid(True)
    # 显示图形
    plt.savefig(os.path.join(output_dir, 'ttt.png'))
    # 将ttt_df和result_df保存到csv文件
    ttt_df.rename(columns={'max': 'leave_time(s)'}, inplace=True)
    ttt_df.to_csv(os.path.join(output_dir, 'ttt.csv'), index=False)
    result_df.to_csv(os.path.join(output_dir, 'ttt_analysis.csv'), index=False)



if __name__ == '__main__':
    path = r'D:\myscripts\pro\output\model3_20250627-140409\trajectory.csv'
    df = pd.read_csv(path)
    plot_ttt(df, window_size=30, output_dir=os.path.dirname(path))
