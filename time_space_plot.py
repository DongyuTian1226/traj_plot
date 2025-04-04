import os
import pandas as pd
import matplotlib.cm as cm
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import Union


class ResearchPlt:
    '''
    更符合科研作图的plt画图, 使用方式：
    1. 继承该类(推荐)
    2. 在对应代码文件开头处, 导入并运行该类, 实现全局配置
    '''
    def __init__(
            self,
            backend='agg',
            figsize=(12, 8),
            dpi=300,
            bbox_inches='tight',
            font='Times New Roman',
            font_size=16,
            legend_framealpha=0.25,
            legend_loc='upper right',
            legend_handletextpad=0,
            ):
        '''设置plt画图的全局参数'''
        # 非交互式模式, 绘图速度更快
        plt.switch_backend(backend)
        # 画布
        plt.rcParams['figure.figsize'] = figsize        # 画布大小
        plt.rcParams['figure.dpi'] = dpi                # 画布分辨率
        plt.rcParams['savefig.bbox'] = bbox_inches      # 保存图像时, 去掉多余空白
        # 字体
        plt.rcParams['font.sans-serif'] = [font]        # 字体
        plt.rcParams['font.size'] = font_size           # 字体大小
        plt.rcParams['axes.unicode_minus'] = False      # 解决负号'-'显示为方块的问题
        # legend
        plt.rcParams['legend.framealpha'] = legend_framealpha           # legend透明
        plt.rcParams['legend.loc'] = legend_loc                         # legend位置
        plt.rcParams['legend.handletextpad'] = legend_handletextpad     # legend图例文字间距

    def show_legend_sorted(self, title: str):
        '''
        处理legend的显示顺序, 使其按照顺序排列
        input
        -----
        title: str, legend标题
        '''
        handles, labels = plt.gca().get_legend_handles_labels()
        sorted_labels = sorted(labels)
        sorted_handles = [handles[labels.index(label)] for label in sorted_labels]
        legend = plt.legend(sorted_handles, sorted_labels, title=title)


# 数据表列索引
# example
# LaneIndex = 1
# CarIDIndex = 0
# FrameIndex = 2
# LocationIndex = 3
# vIndex = 7


class TimeSpacePlotter(ResearchPlt):
    '''时空轨迹图, 继承自ResearchPlt类
    
    时空轨迹图的横坐标为帧号/时间, 纵坐标为车辆(沿行驶方向)位置, 图中颜色表示车辆速度。
    时空轨迹图将按lane画图, 每个lane的车辆轨迹将分别画在同一张图中。
    '''
    def __init__(
            self,
            csv_path: str,
            output_dir: str,
            lane_idx: Union[int, str],
            car_idx: Union[int, str],
            time_idx: Union[int, str],
            distance_idx: Union[int, str],
            v_idx: Union[int, str],
            max_time: int = 1e10,
            v_trans: bool = False,
            colormap: cm = cm.jet_r,
            scatter_size: int = 1,
            **kwargs,
            ):
        '''读取并预处理数据

        input
        -----
        csv_path: str, 仿真数据csv文件路径
        output_dir: str, 保存图片的文件夹
        lane_idx: Union[int, str], 车道号列索引或列名
        car_idx: Union[int, str], 车辆ID列索引或列名
        time_idx: Union[int, str], 帧号列索引或列名
        distance_idx: Union[int, str], 车辆位置列索引或列名
        v_idx: Union[int, str], 车辆速度列索引或列名
        max_time: int, 最大画图的帧数/秒数, 默认值为尽可能大的数字, 即画图范围不限
        v_trans: bool, 是否转换速度单位, 默认False, 即速度单位为m/s, 设为True则转换为km/h
        colormap: cm, 画图参数，颜色映射, 默认为逆序的彩虹色rainbow_r
        scatter_size: int, 画图参数，轨迹的散点大小, 默认为1
        '''
        super().__init__(**kwargs)
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.max_time = max_time
        self.v_trans = v_trans
        self.colormap = colormap
        self.scatter_size = scatter_size
        # 读取数据
        self.data = pd.read_csv(csv_path)
        self.lane_idx = lane_idx if isinstance(lane_idx, str) else self.data.columns[lane_idx]
        self.car_idx = car_idx if isinstance(car_idx, str) else self.data.columns[car_idx]
        self.time_idx = time_idx if isinstance(time_idx, str) else self.data.columns[time_idx]
        self.distance_idx = distance_idx if isinstance(distance_idx, str) else self.data.columns[distance_idx]
        self.v_idx = v_idx if isinstance(v_idx, str) else self.data.columns[v_idx]
        self.data = self.data.sort_values(by=[self.lane_idx, self.car_idx, self.time_idx],
                                          axis=0, ascending=[True, True, True])
        self.data = self.data.reset_index(drop=True)

    def run(self, 
            x_min: int = None, x_max: int = None, x_gap: int = None,
            y_min: int = None, y_max: int = None, y_gap: int = None,
            ):
        '''根据初始化的参数画出时空轨迹图
        
        input
        -----
        x_min: int, 横坐标最小值, 默认None, 即不设置
        x_max: int, 横坐标最大值, 默认None, 即不设置
        x_gap: int, 横坐标前后的空隔, 默认None, 即不设置
        y_min: int, 纵坐标最小值, 默认None, 即不设置
        y_max: int, 纵坐标最大值, 默认None, 即不设置
        y_gap: int, 纵坐标前后的空隔, 默认None, 即不设置
        '''
        # 数据预处理
        data = self.data[self.data[self.time_idx] < self.max_time]
        data[self.v_idx] = data[self.v_idx] * 3.6 if self.v_trans else data[self.v_idx]
        # 画图
        print("begin drawing!")
        for lane, laneGroup in tqdm(data.groupby(data[self.lane_idx])):
            plt.figure()
            for car_id, car_traj in laneGroup.groupby(laneGroup[self.car_idx]):
                car_traj[self.v_idx] = car_traj[self.v_idx].abs() # 速度取绝对值，以免速度方向与指定方向相反而带有负号
                plt.scatter(car_traj[self.time_idx], car_traj[self.distance_idx],
                            c=list(car_traj[self.v_idx]), cmap=self.colormap, s=self.scatter_size)
            plt.title("lane %d"%lane)
            plt.colorbar()
            plt.xlabel(self.time_idx)
            plt.ylabel(self.distance_idx)
            if x_min and x_max:
                x_gap = x_gap or 0
                plt.xlim(x_min - x_gap, x_max + x_gap)
            if y_min and y_max:
                y_gap = y_gap or 0
                plt.ylim(y_min - y_gap, y_max + y_gap)
            plt.savefig(os.path.join(self.output_dir, f"lane_{lane}.jpg"))
            plt.close()
        print("finish drawing!")


def main_sumo_model0():
    path = r'D:\myscripts\sumo\output\test0_post.csv'
    # 创建输出文件夹
    output_manager_dir = path.strip('.csv')
    if not os.path.exists(output_manager_dir):
        os.makedirs(output_manager_dir)
    lane_change_output_dir = os.path.join(output_manager_dir, f'trajectory')
    if not os.path.exists(lane_change_output_dir):
        os.makedirs(lane_change_output_dir)
    # 数据表列索引
    LaneIndex = 2
    CarIDIndex = 0
    TimeIndex = 1
    LocationIndex = 8
    vIndex = 9
    # 运行
    tsp = TimeSpacePlotter(
        csv_path=path, output_dir=lane_change_output_dir,
        lane_idx=LaneIndex, car_idx=CarIDIndex, time_idx=TimeIndex,
        distance_idx=LocationIndex, v_idx=vIndex,
        v_trans=True, scatter_size=2,
        figsize=(20,8),
        )
    tsp.run(y_min=0, y_max=2500, y_gap=50)


if __name__ == '__main__':
    main_sumo_model0()
