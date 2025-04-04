'''画出每辆车的换道情况, 以便确认换道的正确性和换道位置'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
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

    def xy_limit_with_gap(
            self,
            x_min: float = None, x_max: float = None, x_gap: float = None,
            y_min: float = None, y_max: float = None, y_gap: float = None,
            ):
        '''
        设置x轴和y轴的范围, 以及前后给画图留出一定间隔。最终图像呈现的范围为
        [x_min - x_gap, x_max + x_gap]和[y_min - y_gap, y_max + y_gap]

        input
        -----
        x_min, x_max: float, x轴范围
        x_gap: float, x轴范围前后间隔
        y_min, y_max: float, y轴范围
        y_gap: float, y轴范围前后间隔
        '''
        if x_min and x_max:
            x_gap = x_gap or 0
            plt.xlim(x_min - x_gap, x_max + x_gap)
        if y_min and y_max:
            y_gap = y_gap or 0
            plt.ylim(y_min - y_gap, y_max + y_gap)

    def specified_grid(
            self,
            x_grid: list = None, x_grid_color: str = 'grey', x_grid_style: str = '--', x_grid_width: float = 0.5,
            y_grid: list = None, y_grid_color: str = 'black', y_grid_style: str = '-', y_grid_width: float = 0.5,
            ):
        '''
        设置指定的网格线, 颜色, 样式, 宽度
        input
        -----
        x_grid: list, 竖线网格线的x位置
        y_grid: list, 横线网格线的y位置
        x_grid_color, y_grid_color: str, 网格线颜色
        x_grid_style, y_grid_style: str, 网格线样式
        x_grid_width, y_grid_width: float, 网格线宽度
        '''
        for vertical_line in x_grid or []:
            plt.axvline(x=vertical_line, color=x_grid_color, linestyle=x_grid_style, linewidth=x_grid_width)
        for horizontal_line in y_grid or []:
            plt.axhline(y=horizontal_line, color=y_grid_color, linestyle=y_grid_style, linewidth=y_grid_width)


class LaneChangePlot(ResearchPlt):
    '''画出车道变化情况, 适配字符串laneID和整数ID'''
    def __init__(self,
                 csv_path: str,
                 lane_idx: Union[int, str],
                 car_idx: Union[int, str],
                 time_idx: Union[int, str],
                 distance_idx: Union[int, str],
                 x_idx: Union[int, str],
                 y_idx: Union[int, str],
                 lanemode: str = 'legend',
                 **kwargs):
        '''预存储文件存储信息和画图参数

        input
        -----
        csv_path: str, 仿真数据csv文件路径
        lane_idx: Union[int, str], 车道索引
        car_idx: Union[int, str], 车辆索引
        time_idx: Union[int, str], 时间索引
        distance_idx: Union[int, str], 行驶距离索引
        x_idx: Union[int, str], 二维坐标系下x坐标索引
        y_idx: Union[int, str], 二维坐标系下y坐标索引
        lanemode: str, 可选y, legend
        '''
        super().__init__(**kwargs)
        self.csv_path = csv_path
        self.lane_idx = lane_idx
        self.car_idx = car_idx
        self.time_idx = time_idx
        self.distance_idx = distance_idx
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.lanemode = lanemode
        self.df = pd.read_csv(csv_path)
        # 生成存储文件夹
        output_manager_dir = csv_path.strip('.csv')
        if not os.path.exists(output_manager_dir):
            os.makedirs(output_manager_dir)
        lane_change_output_dir = os.path.join(output_manager_dir, f'lane_change_{lanemode}')
        if not os.path.exists(lane_change_output_dir):
            os.makedirs(lane_change_output_dir)
        self.lane_change_output_dir = lane_change_output_dir
        # 车道颜色映射
        # 获取所有唯一的 laneID
        unique_lanes = self.df[self.lane_idx].unique()
        # 创建颜色映射
        self.color_map = {lane: cm.tab10(i) for i, lane in enumerate(unique_lanes)}

    def run(self,
            x_min: int = None, x_max: int = None, x_gap: int = None,
            y_min: int = None, y_max: int = None, y_gap: int = None, y_min_ramp: int = None,
            lane_min: int = None, lane_max: int = None,
            x_grid: list = None, x_grid_color: str = 'grey', x_grid_style: str = '--', x_grid_width: float = 0.5,
            y_grid: list = None, y_grid_color: str = 'black', y_grid_style: str = '-', y_grid_width: float = 0.5,
            **kwargs,
            ):
        '''画出每辆车的换道情况, 以便确认换道的正确性和换道位置
        '''
        for vehicle_id, group in tqdm(self.df.groupby(self.car_idx)):
            self._plot(
                group, 
                x_min=x_min, x_max=x_max, x_gap=x_gap,
                y_min=y_min, y_max=y_max, y_gap=y_gap, y_min_ramp=y_min_ramp,
                lane_min=lane_min, lane_max=lane_max,
                x_grid=x_grid, x_grid_color=x_grid_color, x_grid_style=x_grid_style, x_grid_width=x_grid_width,
                y_grid=y_grid, y_grid_color=y_grid_color, y_grid_style=y_grid_style, y_grid_width=y_grid_width,
                **kwargs,
                )

    def _plot(self, df: pd.DataFrame,
              x_min: int = None, x_max: int = None, x_gap: int = None,
              y_min: int = None, y_max: int = None, y_gap: int = None, y_min_ramp: int = None,
              lane_min: int = None, lane_max: int = None,
              x_grid: list = None, x_grid_color: str = 'grey', x_grid_style: str = '--', x_grid_width: float = 0.5,
              y_grid: list = None, y_grid_color: str = 'black', y_grid_style: str = '-', y_grid_width: float = 0.5,
              ):
        '''画出单车的换道情况, 外部调用推荐run()

        input
        -----
        df: pd.DataFrame, 单车的轨迹数据
        x_min, x_max: int, 横坐标最小值, 横坐标最大值
        x_gap: int, 横坐标前后的空隔
        y_min, y_max: int, 纵坐标最小值, 纵坐标最大值
        y_gap: int, 纵坐标前后的空隔
        y_min_ramp: int, 纵坐标最小值(匝道场景)
        lane_min, lane_max: int, 车道最小值, 车道最大值
        x_grid, y_grid: list, x或y对应垂线的网格线位置
        x_grid_color, y_grid_color: str, 网格线颜色
        x_grid_style, y_grid_style: str, 网格线样式
        x_grid_width, y_grid_width: float, 网格线宽度
        '''
        df = df.sort_values(by=self.time_idx, ascending=True)
        plt.figure()
        # 横轴为distance, 纵轴为laneID
        if self.lanemode == 'y':
            plt.plot(df[self.distance_idx], df[self.lane_idx], label=self.lane_idx)
            plt.xlabel(self.distance_idx)
            plt.ylabel(self.lane_idx)
            if lane_min and lane_max:
                plt.ylim(lane_min, lane_max)
        # 横纵轴为xy, laneID为label
        elif self.lanemode == 'legend':
            for lane in df[self.lane_idx].unique():
                lane_data = df[df[self.lane_idx] == lane]
                plt.scatter(lane_data[self.x_idx], lane_data[self.y_idx],
                            label=lane, color=self.color_map[lane], s=2)
            plt.xlabel(self.x_idx)
            plt.ylabel(self.y_idx)
            self.show_legend_sorted(self.lane_idx)
            if y_min and y_max and max(df[self.y_idx]) - min(df[self.y_idx]) < y_max - y_min:
                self.xy_limit_with_gap(y_min=y_min, y_max=y_max, y_gap=y_gap)
            elif y_min_ramp and y_max:
                self.xy_limit_with_gap(y_min=y_min_ramp, y_max=y_max, y_gap=y_gap)

        # 全局plt配置
        self.xy_limit_with_gap(x_min=x_min, x_max=x_max, x_gap=x_gap)
        # 网格线
        self.specified_grid(x_grid=x_grid, x_grid_color=x_grid_color, x_grid_style=x_grid_style, x_grid_width=x_grid_width,
                            y_grid=y_grid, y_grid_color=y_grid_color, y_grid_style=y_grid_style, y_grid_width=y_grid_width)
        plt.title(f'Vehicle {df[self.car_idx].iloc[0]} Lane Change')
        plt.savefig(os.path.join(self.lane_change_output_dir, f'{df[self.car_idx].iloc[0]}.png'))
        plt.close()


def main():
    # 数据表参数
    car_idx = 'VehicleID'
    time_idx = 'Time(s)'
    lane_idx = 'LaneID'
    x_idx = 'x(m)'
    y_idx = 'y(m)'
    distance_idx = 'distance(m)'
    # 画图参数
    figsize = (12, 4)
    x_min = -1000
    x_max = 1500
    y_min = -20
    y_min_ramp = -52
    y_max = 0
    lane_min = -1
    lane_max = 7
    x_grid = [-1000, 0, 500, 1500]
    y_grid = [x for x in np.arange(0, -20, -3.75)]

    path = r'D:\myscripts\sumo\output\test0_post.csv'
    lcp = LaneChangePlot(
        path,
        lane_idx=lane_idx, car_idx=car_idx,
        time_idx=time_idx, distance_idx=distance_idx,
        x_idx=x_idx, y_idx=y_idx,
        lanemode='legend',
        figsize=figsize,
        )
    lcp.run(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max, y_min_ramp=y_min_ramp,
        lane_min=lane_min, lane_max=lane_max,
        x_grid=x_grid, y_grid=y_grid,
        )


if __name__ == "__main__":
    main()
