'''画出每辆车的换道情况, 以便确认换道的正确性和换道位置'''
import os
from typing import Union
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm

from .research_plt import ResearchPlt
from scipy.stats import gaussian_kde


class LaneChangePlot(ResearchPlt):
    '''画出车道变化情况, 适配字符串laneID和整数ID'''
    def __init__(self,
                 time_idx: Union[int, str],
                 car_idx: Union[int, str],
                 lane_idx: Union[int, str],
                 dist_idx: Union[int, str],
                 x_idx: Union[int, str],
                 y_idx: Union[int, str],
                 data_path: str = None,
                 df: pd.DataFrame = None,
                 save_dir: str = None,
                 max_time: int = 1e10,
                 ids: Union[int, str, list] = None,
                 lanemode: str = 'legend',
                 **kwargs):
        '''预存储文件存储信息和画图参数

        input
        -----
        path: str, 仿真数据csv文件路径
        time_idx: Union[int, str], 时间索引
        car_idx: Union[int, str], 车辆索引
        lane_idx: Union[int, str], 车道索引
        dist_idx: Union[int, str], 行驶距离索引
        x_idx: Union[int, str], 二维坐标系下x坐标索引
        y_idx: Union[int, str], 二维坐标系下y坐标索引
        data_path: str, 数据路径, 与df必须提供二者之一
        df: pd.DataFrame, 数据, 与data_path必须提供二者之一
        save_dir: str, 保存路径
        lanemode: str, 可选y, legend
        **kwargs: ResearchPlt的初始化参数, 参见ResearchPlt
        '''
        if (not data_path and df is None) or (data_path and df is not None):
            raise ValueError('data_path and df must provide only one In Calling LaneChangePlot')
        if df is not None and not save_dir:
            raise ValueError('save_dir must provide when df is provided')
        super().__init__(**kwargs)
        self.path = data_path
        # 生成存储文件夹
        save_dir = save_dir or os.path.join(os.path.dirname(data_path), f'lane_change_{lanemode}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.max_time = max_time
        self.ids = ids
        self.lanemode = lanemode
        # 读取数据
        self.df = df if df is not None else (pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_excel(data_path))
        self.lane_idx = lane_idx if isinstance(lane_idx, str) else self.df.columns[lane_idx]
        self.car_idx = car_idx if isinstance(car_idx, str) else self.df.columns[car_idx]
        self.time_idx = time_idx if isinstance(time_idx, str) else self.df.columns[time_idx]
        self.dist_idx = dist_idx if isinstance(dist_idx, str) else self.df.columns[dist_idx]
        self.x_idx = x_idx if isinstance(x_idx, str) else self.df.columns[x_idx]
        self.y_idx = y_idx if isinstance(y_idx, str) else self.df.columns[y_idx]
        # 数据预处理
        self.df = self.df[self.df[self.time_idx] <= self.max_time]
        if self.ids is not None:
            self.df = self.df[self.df[self.car_idx].isin(self.ids)]
        self.df = self.df.sort_values(by=[self.time_idx, self.car_idx], axis=0, ascending=[True, True])
        self.df = self.df.reset_index(drop=True)

        self._init_lane_color_map()

    def run(self,
            x_min: int = None, x_max: int = None, x_gap: int = 0,
            y_min: int = None, y_max: int = None, y_gap: int = 0, y_min_ramp: int = None,
            lane_min: int = None, lane_max: int = None,
            x_grid: list = None, x_grid_color: str = 'grey', x_grid_style: str = '--', x_grid_width: float = 0.5,
            y_grid: list = None, y_grid_color: str = 'black', y_grid_style: str = '-', y_grid_width: float = 0.5,
            line_width: float = 1, line_style: str = '-',
            marker: str = 'o', markersize: float = 1, marker_alpha: float = 0.5,
            ):
        '''画出每辆车的换道情况, 以便确认换道的正确性和换道位置
        '''
        for _, group in tqdm(self.df.groupby(self.car_idx)):
            self._plot(
                group,
                x_min=x_min, x_max=x_max, x_gap=x_gap,
                y_min=y_min, y_max=y_max, y_gap=y_gap, y_min_ramp=y_min_ramp,
                lane_min=lane_min, lane_max=lane_max,
                x_grid=x_grid, x_grid_color=x_grid_color, x_grid_style=x_grid_style, x_grid_width=x_grid_width,
                y_grid=y_grid, y_grid_color=y_grid_color, y_grid_style=y_grid_style, y_grid_width=y_grid_width,
                line_width=line_width, line_style=line_style,
                marker=marker, markersize=markersize, marker_alpha=marker_alpha,
                )

    def _plot(self, car_df: pd.DataFrame,
              x_min: int = None, x_max: int = None, x_gap: int = 0,
              y_min: int = None, y_max: int = None, y_gap: int = 0, y_min_ramp: int = None,
              lane_min: int = None, lane_max: int = None,
              x_grid: list = None, x_grid_color: str = 'grey', x_grid_style: str = '--', x_grid_width: float = 0.5,
              y_grid: list = None, y_grid_color: str = 'black', y_grid_style: str = '-', y_grid_width: float = 0.5,
              line_width: float = 1, line_style: str = '-',
              marker: str = 'o', markersize: float = 1, marker_alpha: float = 0.5,
              ):
        '''画出单车的换道情况, 外部调用推荐run()

        input
        -----
        car_df: pd.DataFrame, 单车的轨迹数据
        x_min, x_max: int, 横坐标最小值, 横坐标最大值
        x_gap: int, 横坐标前后的空隔
        y_min, y_max: int, 纵坐标最小值, 纵坐标最大值
        y_gap: int, 纵坐标前后的空隔
        y_min_ramp: int, 纵坐标最小值(匝道场景)
        lane_min, lane_max: int, 车道最小值, 车道最大值, 当纵坐标为lane数值考虑设置
        x_grid, y_grid: list, x或y对应垂线的网格线位置
        x_grid_color, y_grid_color: str, 网格线颜色
        x_grid_style, y_grid_style: str, 网格线样式
        x_grid_width, y_grid_width: float, 网格线宽度
        line_width, line_style: float, str, 线的宽度和样式
        marker, markersize, marker_alpha: str, float, float, 点的样式、大小和透明度
        '''
        plt.figure()
        # 横轴为distance, 纵轴为laneID
        if self.lanemode == 'y':
            plt.plot(
                car_df[self.dist_idx], car_df[self.lane_idx], label=self.lane_idx,
                line_width=line_width, line_style=line_style)
            plt.xlabel(self.dist_idx)
            plt.ylabel(self.lane_idx)
            if lane_min and lane_max:
                plt.ylim(lane_min, lane_max)
        # 横纵轴为xy, laneID为label
        elif self.lanemode == 'legend':
            for lane, lane_data in car_df.groupby(self.lane_idx):
                plt.scatter(lane_data[self.x_idx], lane_data[self.y_idx],
                            label=lane, color=self.lane_color_map[lane],
                            marker=marker, s=markersize, alpha=marker_alpha)
            plt.xlabel(self.x_idx)
            plt.ylabel(self.y_idx)
            self.show_legend_sorted(self.lane_idx)
            if y_min and y_max and \
                max(car_df[self.y_idx]) - min(car_df[self.y_idx]) < y_max - y_min:
                self.xy_limit_with_gap(y_min=y_min, y_max=y_max, y_gap=y_gap)
            elif y_min_ramp and y_max:
                self.xy_limit_with_gap(y_min=y_min_ramp, y_max=y_max, y_gap=y_gap)

        # 全局plt配置
        self.xy_limit_with_gap(x_min=x_min, x_max=x_max, x_gap=x_gap)
        # 网格线
        self.xy_grid(x_grid=x_grid, x_grid_color=x_grid_color,
                     x_grid_style=x_grid_style, x_grid_width=x_grid_width,
                     y_grid=y_grid, y_grid_color=y_grid_color,
                     y_grid_style=y_grid_style, y_grid_width=y_grid_width)
        car_id = car_df[self.car_idx].iloc[0]
        plt.title(f'Vehicle {car_id} Lane Change')
        plt.savefig(os.path.join(self.save_dir, f'{car_id}.png'))
        plt.close()

    def _init_lane_color_map(self, cmap = cm.tab10):
        '''初始化车道颜色映射。
        如不需要对车道指定颜色, 可关闭。
        '''
        unique_lanes = self.df[self.lane_idx].unique()
        self.lane_color_map = {lane: cmap(i) for i, lane in enumerate(unique_lanes)}


class GlobalLaneChangePlot(ResearchPlt):
    '''全局画图, 所有车辆的轨迹在一张图上'''
    def __init__(self,
                 time_idx: Union[int, str],
                 car_idx: Union[int, str], 
                 lane_idx: Union[int, str],
                 dist_idx: Union[int, str],
                 x_idx: Union[int, str],
                 y_idx: Union[int, str],
                 data_path: str = None,
                 df: pd.DataFrame = None,
                 save_dir: str = None,
                 max_time: int = 1e10,
                 ids: Union[int, str, list] = None,
                 **kwargs):
        '''预存储文件存储信息和画图参数

        input
        -----
        path: str, 仿真数据csv文件路径
        time_idx: Union[int, str], 时间索引
        car_idx: Union[int, str], 车辆索引
        lane_idx: Union[int, str], 车道索引
        dist_idx: Union[int, str], 行驶距离索引
        x_idx: Union[int, str], 二维坐标系下x坐标索引
        y_idx: Union[int, str], 二维坐标系下y坐标索引
        save_dir: str, 保存路径
        max_time: int, 最大时间限制
        ids: Union[int, str, list], 指定车辆ID
        **kwargs: ResearchPlt的初始化参数
        '''
        if (not data_path and df is None) or (data_path and df is not None):
            raise ValueError('data_path and df must provide only one In Calling LaneChangePlot')
        if df is not None and not save_dir:
            raise ValueError('save_dir must provide when df is provided')
        super().__init__(**kwargs)
        self.path = data_path
        # 生成存储文件夹
        save_dir = save_dir or os.path.join(os.path.dirname(data_path), 'global_lane_change')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.max_time = max_time
        self.ids = ids

        # 读取数据
        self.df = df if df is not None else (pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_excel(data_path))
        self.lane_idx = lane_idx if isinstance(lane_idx, str) else self.df.columns[lane_idx]
        self.car_idx = car_idx if isinstance(car_idx, str) else self.df.columns[car_idx]
        self.time_idx = time_idx if isinstance(time_idx, str) else self.df.columns[time_idx]
        self.dist_idx = dist_idx if isinstance(dist_idx, str) else self.df.columns[dist_idx]
        self.x_idx = x_idx if isinstance(x_idx, str) else self.df.columns[x_idx]
        self.y_idx = y_idx if isinstance(y_idx, str) else self.df.columns[y_idx]

        # 数据预处理
        self.df = self.df[self.df[self.time_idx] <= self.max_time]
        if self.ids is not None:
            self.df = self.df[self.df[self.car_idx].isin(self.ids)]
        self.df = self.df.sort_values(by=[self.time_idx, self.car_idx], axis=0, ascending=[True, True])
        self.df = self.df.reset_index(drop=True)

    def run(self,
            x_min: int = None, x_max: int = None, x_gap: int = 0, x_offset: int = 0,
            y_min: int = None, y_max: int = None, y_gap: int = 0, y_offset: int = 0,
            lane_min: int = None, lane_max: int = None,
            x_grid: list = None, x_grid_color: str = 'grey', x_grid_style: str = '--', x_grid_width: float = 0.5,
            y_grid: list = None, y_grid_color: str = 'black', y_grid_style: str = '-', y_grid_width: float = 0.5,
            plot_type: str = 'heatmap',  # 可选 'heatmap', 'log_heatmap', 'scatter', 'direction_scatter', 'aggregate_direction'
            cmap: str = 'YlOrRd',
            x_bins: int = 50,  # x方向的网格数
            lanemode: str = 'y',
            scatter_size_range: tuple = (20, 200),  # 散点图尺寸范围
            scatter_alpha_range: tuple = (0.1, 0.8),  # 散点图透明度范围
            arrow_scale: float = 50,  # 方向箭头的缩放比例
            ):
        '''画出全局换道情况

        input
        -----
        plot_type: str, 绘图类型, 可选'heatmap'(热力图), 'log_heatmap'(对数热力图), 'scatter'(散点图), 'direction_scatter'(带方向的散点图)
        scatter_size_range: tuple, 散点图尺寸范围
        scatter_alpha_range: tuple, 散点图透明度范围
        arrow_scale: float, 方向箭头的缩放比例
        其他参数同原函数
        '''
        # 数据预处理
        self.df[self.x_idx] = self.df[self.x_idx] + x_offset
        self.df[self.y_idx] = self.df[self.y_idx] + y_offset

        # 找出所有换道点及方向
        lane_changes = []
        for _, car_data in self.df.groupby(self.car_idx):
            lane_shifts = car_data[self.lane_idx].diff()
            change_points = car_data[lane_shifts != 0].copy()
            if not change_points.empty:
                # 记录换道方向
                change_points['direction'] = lane_shifts[lane_shifts != 0]
                lane_changes.append(change_points)

        if not lane_changes:
            print('No lane changes found.')
            return

        lane_changes_df = pd.concat(lane_changes)

        # 创建画布
        plt.figure()

        # 根据lanemode选择坐标轴
        if lanemode == 'y':
            x = lane_changes_df[self.x_idx]
            y = lane_changes_df[self.lane_idx]
            xlabel = self.dist_idx
            ylabel = self.lane_idx
            unique_lanes = sorted(lane_changes_df[self.lane_idx].unique())
            y_edges = np.arange(min(unique_lanes) - 0.5, max(unique_lanes) + 1.5)
        else:
            x = lane_changes_df[self.x_idx]
            y = lane_changes_df[self.y_idx]
            xlabel = self.x_idx
            ylabel = self.y_idx
            y_edges = np.linspace(y_min or y.min(), y_max or y.max(), x_bins + 1)

        x_edges = np.linspace(x_min or x.min(), x_max or x.max(), x_bins + 1)

        if plot_type in ['heatmap', 'log_heatmap']:
            self._plot_heatmap(x, y, x_edges, y_edges, cmap, 
                             log_scale=(plot_type == 'log_heatmap'))
        elif plot_type == 'scatter':
            self._plot_density_scatter(x, y, x_bins, scatter_size_range, 
                                     scatter_alpha_range)
        elif plot_type == 'direction_scatter':
            self._plot_direction_scatter(x, y, lane_changes_df['direction'], 
                                       x_bins, scatter_size_range, 
                                       scatter_alpha_range, arrow_scale)
        elif plot_type == 'aggregate_direction':
            self._plot_aggregated_direction_scatter(x, y, lane_changes_df['direction'],
                                         x_bins, scatter_size_range,
                                         scatter_alpha_range, arrow_scale)

        # 设置标签和范围
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if lanemode == 'y' and lane_min and lane_max:
            plt.ylim(lane_min, lane_max)

        # 设置坐标轴和网格
        self.xy_limit_with_gap(x_min=x_min, x_max=x_max, x_gap=x_gap,
                              y_min=y_min, y_max=y_max, y_gap=y_gap)
        self.xy_grid(x_grid=x_grid, x_grid_color=x_grid_color,
                    x_grid_style=x_grid_style, x_grid_width=x_grid_width,
                    y_grid=y_grid, y_grid_color=y_grid_color,
                    y_grid_style=y_grid_style, y_grid_width=y_grid_width)

        plt.title('Global Lane Change Distribution')
        plt.savefig(os.path.join(self.save_dir, f'global_lane_changes_{plot_type}.png'))
        plt.close()

    def _plot_heatmap(self, x, y, x_edges, y_edges, cmap, log_scale=False):
        '''绘制热力图'''
        H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        if log_scale:
            H = np.log1p(H)  # log1p处理0值
        plt.pcolormesh(x_edges, y_edges, H.T, cmap=cmap)
        plt.colorbar(label='Lane Change Count (log)' if log_scale else 'Lane Change Count')

    def _plot_density_scatter(self, x, y, bins, size_range, alpha_range):
        '''绘制密度散点图'''
        # 计算每个点的局部密度
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        
        # 归一化密度值到指定范围
        sizes = np.interp(z, (z.min(), z.max()), size_range)
        alphas = np.interp(z, (z.min(), z.max()), alpha_range)
        
        # 绘制散点图
        plt.scatter(x, y, s=sizes, alpha=alphas, c=z, cmap='viridis')
        plt.colorbar(label='Density')

    def _plot_direction_scatter(self, x, y, directions, bins, size_range, alpha_range, arrow_scale):
        '''绘制带方向的散点图'''
        # 计算每个点的局部密度
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        
        # 归一化密度值到指定范围
        sizes = np.interp(z, (z.min(), z.max()), size_range)
        alphas = np.interp(z, (z.min(), z.max()), alpha_range)

        # 添加方向箭头
        # 计算箭头的起点和终点
        dx = np.zeros_like(x)  # x方向无位移
        dy = directions * arrow_scale  # y方向位移与换道方向成正比
        
        # 绘制箭头
        plt.quiver(x, y, dx, dy, 
                  directions,  # 箭头颜色与方向一致
                  cmap='RdYlBu',
                  scale=arrow_scale*20,  # 调整箭头大小
                  width=0.003,  # 箭头宽度
                  alpha=0.5)  # 箭头透明度

        # 绘制散点图
        scatter = plt.scatter(x, y, s=sizes, alpha=alphas, c=directions, cmap='viridis')
        plt.colorbar(scatter, label='Lane Change Direction')

    def _plot_aggregated_direction_scatter(self, x, y, directions, bins, size_range, alpha_range, arrow_scale):
        '''绘制聚合后的方向散点图'''
        # 创建网格
        x_edges = np.linspace(min(x), max(x), bins)
        y_edges = np.linspace(min(y), max(y), bins)
        
        # 初始化聚合数据存储
        grid_counts = np.zeros((bins-1, bins-1))  # 每个网格的换道数量
        grid_directions = np.zeros((bins-1, bins-1))  # 每个网格的平均换道方向
        
        # 计算网格中心点坐标
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        
        # 聚合数据
        for i in range(len(x_edges)-1):
            for j in range(len(y_edges)-1):
                # 找出落在当前网格的点
                mask = (x >= x_edges[i]) & (x < x_edges[i+1]) & \
                       (y >= y_edges[j]) & (y < y_edges[j+1])
                grid_counts[i,j] = np.sum(mask)
                if grid_counts[i,j] > 0:
                    # 计算该网格内的平均换道方向
                    grid_directions[i,j] = np.mean(directions[mask])
        
        # 对换道数量取对数,避免数值差异过大
        log_counts = np.log1p(grid_counts)
        
        # 归一化箭头参数
        normalized_counts = np.interp(log_counts, 
                                    (log_counts.min(), log_counts.max()), 
                                    (0.001, 1.0))
        
        # 绘制聚合后的箭头
        for i in range(len(x_centers)):
            for j in range(len(y_centers)):
                if grid_counts[i,j] > 0:
                    # 箭头宽度和长度基于对数密度变化
                    width = normalized_counts[i,j] * 0.015  # 增大基础宽度
                    length = normalized_counts[i,j] * arrow_scale
                    alpha = normalized_counts[i,j] * 0.6 + 0.4  # 提高最小透明度
                    
                    # 绘制箭头
                    plt.quiver(x_centers[i], y_centers[j], 
                             0,  # x方向无位移
                             grid_directions[i,j] * length,  # y方向位移随密度变化
                             grid_directions[i,j],  # 箭头颜色
                             cmap='RdYlBu',
                             scale=arrow_scale*10,
                             width=width,
                             alpha=alpha)
        
        # 添加密度热力图背景
        plt.hist2d(x, y, bins=[x_edges, y_edges], 
                  cmap='YlOrRd', alpha=0.3,
                  norm=plt.matplotlib.colors.LogNorm())  # 使用对数归一化
        plt.colorbar(label='Lane Change Density (log scale)')


def main():
    '''研究生毕设sumo仿真结果画图'''
    # 数据表参数
    car_idx = 'vehicleID'
    time_idx = 'time(s)'
    lane_idx = 'laneID'
    x_idx = 'x(m)'
    y_idx = 'y(m)'
    dist_idx = 'distance(m)'
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
    y_grid = list(np.arange(0, -20, -3.75))

    path = r'D:\myscripts\pro\output\model0\trajectory.csv'
    lcp = LaneChangePlot(
        path,
        lane_idx=lane_idx, car_idx=car_idx,
        time_idx=time_idx, dist_idx=dist_idx,
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


def global_main():
    '''画整个轨迹数据集的换道图'''
    data_path = r'D:\myscripts\pro\output\model1_20250609-204128\trajectory.csv'
    time_idx = 'time(s)'
    car_idx = 'vehicleID'
    lane_idx = 'laneID'
    dist_idx = 'distance(m)'
    x_idx = 'x(m)'
    y_idx = 'y(m)'
    gcp = GlobalLaneChangePlot(
        data_path,
        time_idx=time_idx, car_idx=car_idx, dist_idx=dist_idx,
        lane_idx=lane_idx, x_idx=x_idx, y_idx=y_idx,
        )
    gcp.run(
        x_min=0, x_max=1400, x_offset=1000,
        # plot_type='heatmap',
        # plot_type='log_heatmap',
        # plot_type='scatter',
        # plot_type='direction_scatter',
        plot_type='aggregate_direction',
    )


if __name__ == "__main__":
    # main()
    global_main()
