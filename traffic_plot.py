'''对数据进行聚合等操作后, 对交通数据进行画图'''
import os
from math import nan
import pandas as pd
import matplotlib.pyplot as plt
from .research_plt import ResearchPlt


def section_v_hills(
    df: pd.DataFrame, output_dir: str, window_idx: str, window_size: int, window_step: int, time_step: int = 200,
    xlines: list = None, time_idx = 'time(s)', lane_idx = 'laneID', speed_idx = 'speed(m/s)',
) -> str:
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
    time_step: 时间步长, 每隔多少秒进行一次统计, 默认为200s
    xlines: 标记线, 用于在指定的横轴位置标记竖线

    return
    ------
    save_path: 输出图片的保存路径
    '''
    # 时间起止
    time_start, time_end = df[time_idx].min(), df[time_idx].max()
    time_num = int((time_end - time_start) / time_step)   # 时间窗口数量
    # 窗口起止
    window_start, window_end = df[window_idx].min(), df[window_idx].max()
    window_num = int((window_end - window_start) / window_step)
    window_centers = [window_start + i * window_step + window_size / 2 for i in range(window_num + 1)]
    # 统计类别
    lanes = sorted(df[lane_idx].unique())

    # 绘图
    ResearchPlt()
    fig, axes = plt.subplots(figsize=(8, 12), nrows=time_num, ncols=1, sharex=True)
    ymax = -1e6  # 用于记录每个时间窗口的最大y值, 以便统一y轴范围
    for time_k in range(1, time_num + 1):
        # 当前时间数据
        t = time_k * time_step + time_start
        cur_df = df[df[time_idx] == t]
        if cur_df.empty:
            print(f'No data at time {t} in ploting section_v_hills')
            continue
        # 以步长遍历窗口, 统计数据
        statistics = {x: [] for x in lanes}  # 按车道ID统计
        for w in window_centers:
            # 窗口数据
            window_data = cur_df[
                (cur_df[window_idx] >= w - window_size / 2) & (cur_df[window_idx] < w + window_size / 2)]
            # 使用groupby按车道统计平均速度
            window_values = window_data.groupby(lane_idx)[speed_idx].mean() * 3.6   # 转换为km/h
            for lane in lanes:
                statistics[lane].append(window_values.get(lane, 0))
            ymax = max(ymax, window_values.max() if not window_values.empty else 0)
        # 绘制当前时间的ttc空间分布图
        plt.subplot(time_num, 1, time_k)
        for lane, values in statistics.items():
            if not values:
                continue
            if time_k == 1:
                plt.plot(window_centers, values, label=f'Lane {lane}')
            else:
                plt.plot(window_centers, values)    # 仅生成一组label
            plt.fill_between(window_centers, values, alpha=0.1)
        plt.xlim(window_start, window_end)
    # 统一子图元素
    for i, ax in enumerate(axes):
        ax.set_ylim(0, ymax * 1.1)
        ax.text(0.01, 0.9, f'{(i + 1) * time_step}s', transform=ax.transAxes,
                fontsize=14, verticalalignment='center', horizontalalignment='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if xlines:
            for x in xlines:
                ax.axvline(x=x, color='r', linestyle='--', linewidth=0.5)  # 添加竖线标记
    # 设置全局标题和标签
    plt.suptitle('Velocity Distribution Across Time and Space')
    # 设置全局xy轴标签
    fig.text(0.5, 0.04, window_idx, ha='center', va='center')
    fig.text(0.06, 0.5, 'section velocity(km/h)', ha='center', va='center', rotation='vertical')
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(lanes))
    fig.legend(loc='upper center', bbox_to_anchor=(0.1, 0.85, 0.89, 0.1), ncol=len(lanes), mode='expand')
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])  # 调整布局以适应标题和图例
    save_path = os.path.join(output_dir, 'section_v_hills.png')
    plt.savefig(save_path)
    plt.close()
    return save_path


def section_k_hills(
    df: pd.DataFrame, output_dir: str, window_idx: str, window_size: int, window_step: int, time_step: int = 200,
    xlines: list = None, time_idx = 'time(s)', lane_idx = 'laneID',
) -> str:
    '''基于空间的密度时间山峦图
    每张图像中: 横轴为路段距离, 纵轴为统计窗口路段内交通流密度(veh/km)
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
    time_step: 时间步长, 每隔多少秒进行一次统计, 默认为200s
    xlines: 标记线, 用于在指定的横轴位置标记竖线

    return
    ------
    save_path: 输出图片的保存路径
    '''
    # 时间起止
    time_start, time_end = df[time_idx].min(), df[time_idx].max()
    time_num = int((time_end - time_start) / time_step)   # 时间窗口数量
    # 窗口起止
    window_start, window_end = df[window_idx].min(), df[window_idx].max()
    window_num = int((window_end - window_start) / window_step)
    window_centers = [window_start + i * window_step + window_size / 2 for i in range(window_num + 1)]
    # 统计类别
    lanes = sorted(df[lane_idx].unique())

    # 绘图
    ResearchPlt()
    fig, axes = plt.subplots(figsize=(8, 12), nrows=time_num, ncols=1, sharex=True)
    ymax = -1e6  # 用于记录每个时间窗口的最大y值, 以便统一y轴范围
    for time_k in range(1, time_num + 1):
        # 当前时间数据
        t = time_k * time_step + time_start
        cur_df = df[df[time_idx] == t]
        if cur_df.empty:
            continue
        # 以步长遍历窗口, 统计数据
        statistics = {x: [] for x in lanes}  # 按车道ID统计
        for w in window_centers:
            # 窗口数据
            window_data = cur_df[
                (cur_df[window_idx] >= w - window_size / 2) & (cur_df[window_idx] < w + window_size / 2)]
            # 使用groupby按车道统计密度
            window_values = window_data.groupby(lane_idx).size() / window_size *1000 # 计算密度, 单位为辆/km
            for lane in lanes:
                statistics[lane].append(window_values.get(lane, 0))
            ymax = max(ymax, window_values.max() if not window_values.empty else 0)
        # 绘制当前时间的ttc空间分布图
        plt.subplot(time_num, 1, time_k)
        for lane, values in statistics.items():
            if not values:
                continue
            if time_k == 1:
                plt.plot(window_centers, values, label=f'Lane {lane}')
            else:
                plt.plot(window_centers, values)    # 仅生成一组label
            plt.fill_between(window_centers, values, alpha=0.1)
        plt.xlim(window_start, window_end)
    # 统一子图元素
    for i, ax in enumerate(axes):
        ax.set_ylim(0, ymax * 1.1)
        ax.text(0.01, 0.9, f'{(i + 1) * time_step}s', transform=ax.transAxes,
                fontsize=14, verticalalignment='center', horizontalalignment='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if xlines:
            for x in xlines:
                ax.axvline(x=x, color='r', linestyle='--', linewidth=0.5)
    # 设置全局标题和标签
    plt.suptitle('Density Distribution Across Time and Space')
    # 设置全局xy轴标签
    fig.text(0.5, 0.04, window_idx, ha='center', va='center')
    fig.text(0.06, 0.5, 'section density(veh/km)', ha='center', va='center', rotation='vertical')
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(lanes))
    fig.legend(loc='upper center', bbox_to_anchor=(0.1, 0.85, 0.89, 0.1), ncol=len(lanes), mode='expand')
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])      # 调整布局以适应标题和图例
    save_path = os.path.join(output_dir, 'section_k_hills.png')
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_ttc(
    df: pd.DataFrame, output_dir: str, thresholds: list = (1, 2, 3),
    time_idx: str = 'time(s)', ttc_idx: str = 'ttc(s)', aggregation_num: int = 10,
) -> str:
    '''绘制不同ttc阈值的数量情况。
    ttc: time to collision碰撞时间, (gap-length) / (spped-leader_speed), 衡量碰撞风险。
    由于在几乎不相撞情况下, ttc为负数或较大数值, 故不考虑其某一时刻的ttc数据分布情况。

    input
    -----
    df: 数据, 包含车辆的基本信息, 如: 车辆ID, 时间, 车道ID, 位置等, ttc已存在。
    length: 车辆长度, 单位m, 用于计算ttc
    output_dir: 输出文件夹
    thresholds: ttc阈值列表, 单位s
    aggregation_num: 聚合数量, 即每隔aggregation_num行聚合一次, 默认为10

    return
    ------
    save_path: 输出图片的保存路径
    '''
    ResearchPlt(legend_loc='upper left')
    grouped = df.groupby(by=time_idx)
    # 统计不同阈值下的情况
    ttc_counts = []
    for thres in thresholds:
        # 注意, ttc为负数表示无碰撞风险, 因此不纳入统计次数
        tmp = thres
        count = grouped.apply(
            lambda x: ((x[ttc_idx] > 0) & (x[ttc_idx] <= tmp)).sum(),
            include_groups=False
            ).reset_index(name=f'count_ttc<{tmp}s')
        ttc_counts.append(count if not ttc_counts else count.iloc[:, 1])    # 仅保留第一个df表的时间列
    ttc_counts_df = pd.concat(ttc_counts, axis=1)
    # 对ttc_counts重新聚合
    aggregated_ttc_counts = []
    for i in range(0, len(ttc_counts_df), aggregation_num):
        # 取aggregation_num条数据进行聚合
        aggregated_row = ttc_counts_df.iloc[i:i + aggregation_num].mean()
        aggregated_ttc_counts.append(aggregated_row)
    # 创建新的DataFrame
    ttc_counts_df = pd.DataFrame(aggregated_ttc_counts)
    # 可视化
    for thres in thresholds:
        plt.plot(ttc_counts_df[time_idx], ttc_counts_df[f'count_ttc<{thres}s'], label=f'Count TTC < {thres}s')
    # 添加标题和标签
    plt.title('Time to Collision (TTC) Analysis')
    plt.xlabel('Time (s)')
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.ylabel('TTC Count')
    # 添加图例
    plt.legend()
    # 添加网格
    plt.grid(True)
    # 保存图形
    save_path = os.path.join(output_dir, 'ttc_count.png')
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_thw(
    df: pd.DataFrame, output_dir: str, thresholds: list = (1, 2, 3, 4, 5),
    time_idx: str = 'time(s)', thw_idx: str = 'thw(s)', aggregation_num: int = 10,
) -> str:
    '''绘制不同thw阈值的数量情况。
    thw: time headway车头时距, gap / speed, 衡量跟驰安全性。
    由于在基本不会相撞的情况下, thw较大数值不存在过多的实际意义, 故不考虑其某一时刻的thw数据分布情况。

    input
    -----
    df: 数据, 包含车辆的基本信息, 如: 车辆ID, 时间, 车道ID, 位置等
    output_dir: 输出文件夹
    thresholds: thw阈值列表, 单位s
    aggregation_num: 聚合数量, 即每隔aggregation_num行聚合一次, 默认为10

    return
    ------
    save_path: 输出图片的保存路径
    '''
    ResearchPlt(legend_loc='upper left')
    grouped = df.groupby(by=time_idx)
    # 统计不同阈值下的情况
    thw_counts = []
    for thres in thresholds:
        count = grouped.apply(
            lambda x: (x[thw_idx] <= thres).sum(),
            include_groups=False
            ).reset_index(name=f'count_thw<{thres}s')
        thw_counts.append(count if not thw_counts else count.iloc[:, 1])    # 仅保留第一个df表的时间列
    thw_counts_df = pd.concat(thw_counts, axis=1)
    # 对thw_counts重新聚合
    aggregated_thw_counts = []
    for i in range(0, len(thw_counts_df), aggregation_num):
        # 取aggregation_num条数据进行聚合
        aggregated_row = thw_counts_df.iloc[i:i + aggregation_num].mean()
        aggregated_thw_counts.append(aggregated_row)
    # 创建新的DataFrame
    thw_counts_df = pd.DataFrame(aggregated_thw_counts)
    # 可视化
    for thres in thresholds:
        plt.plot(thw_counts_df[time_idx], thw_counts_df[f'count_thw<{thres}s'], label=f'Count Time Headway < {thres}s')
    # 添加标题和标签
    plt.title('Time Headway (THW) Analysis')
    plt.xlabel('Time (s)')
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.ylabel('Time Headway Count')
    # 添加图例
    plt.legend()
    # 添加网格
    plt.grid(True)
    # 保存图形
    save_path = os.path.join(output_dir, 'time_headway_count.png')
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_ttt(       # pylint: disable=too-many-statements
    df: pd.DataFrame, window_size: int, output_dir: str,
    car_idx ='vehicleID', time_idx = 'time(s)',
) -> str:
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

    return
    ------
    save_path: 输出图片的保存路径
    '''
    ResearchPlt(legend_loc='upper left')       # 执行全局科研可视化配置
    # 按车辆ID分组，计算每辆车的总旅行时间
    grouped = df.groupby(car_idx)
    ttt_df = grouped[time_idx].agg(['max', 'min']).reset_index()
    ttt_df['ttt(s)'] = ttt_df['max'] - ttt_df['min']
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
        mainline_window_data = ttt_df[
            (ttt_df['max'] >= t - window_size) & (ttt_df['max'] < t) & (ttt_df['vehicle_type'] == 'mainline')]
        ramp_window_data = ttt_df[
            (ttt_df['max'] >= t - window_size) & (ttt_df['max'] < t) & (ttt_df['vehicle_type'] == 'ramp')]
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
    plt.plot(result_df['time(s)'], result_df['cumulative_avg_all'],
             label='Cumulative Avg All', color='red', linestyle='--')

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
    plt.plot(result_df['time(s)'], result_df['cumulative_avg_mainline'],
             label='Cumulative Avg Mainline', color='darkblue', linestyle='--')
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
    plt.plot(result_df['time(s)'], result_df['cumulative_avg_ramp'],
             label='Cumulative Avg Ramp', color='darkorange', linestyle='--')
    plt.fill_between(result_df['time(s)'],
                     result_df['cumulative_avg_ramp'] - result_df['cumulative_std_ramp'],
                     result_df['cumulative_avg_ramp'] + result_df['cumulative_std_ramp'],
                     color='darkorange', alpha=0.1)
    # 设置标题和标签
    plt.title('Total Travel Time (TTT) Analysis')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Travel Time (s)')
    plt.xlim(0, end_time)
    plt.ylim(0, None)
    plt.legend()
    # 添加网格
    plt.grid(True)
    # 显示图形
    save_path = os.path.join(output_dir, 'ttt.png')
    plt.savefig(save_path)
    plt.close()
    # 将ttt_df和result_df保存到csv文件
    ttt_df.rename(columns={'max': 'leave_time(s)'}, inplace=True)
    ttt_df.to_csv(os.path.join(output_dir, 'ttt.csv'), index=False)
    result_df.to_csv(os.path.join(output_dir, 'ttt_analysis.csv'), index=False)
    return save_path


if __name__ == '__main__':
    PATH = r'D:\myscripts\pro\output\model3_20250627-140409\trajectory.csv'
    data = pd.read_csv(PATH)
    plot_ttt(data, window_size=30, output_dir=os.path.dirname(PATH))
