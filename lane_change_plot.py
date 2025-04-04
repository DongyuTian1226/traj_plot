'''画出每辆车的换道情况, 以便确认换道的正确性和换道位置'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from typing import Union

from research_plt import ResearchPlt


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

    path = r'D:\myscripts\pro\output\test0_post.csv'
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
