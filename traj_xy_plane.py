import os
from typing import Union, Callable
import pandas as pd
from tqdm import tqdm
from matplotlib import cm
import matplotlib.pyplot as plt

from .research_plt import ResearchPlt


class TrajXyPlot(ResearchPlt):
    '''轨迹点在xy平面上绘制, 继承自ResearchPlt类

    # TODO 根据example更新代码
    '''
    def __init__(
            self,
            path: str,
            time_idx: Union[int, str],
            car_idx: Union[int, str],
            x_idx: Union[int, str],
            y_idx: Union[int, str],
            lane_idx: Union[int, str] = None,
            v_idx: Union[int, str] = None,
            save_dir: str = None,
            max_time: int = 1e10,
            ids: Union[int, str, list] = None,
            v_trans: bool = False,
            v_abs: bool = False,
            legend_mode: str = 'lane',
            separate_plot: bool = False,
            **kwargs):
        '''继承自ResearchPlt类, 画轨迹点在xy平面上的散点图

        input
        -----
        path: str, 轨迹文件路径
        time_idx: Union[int, str], 时间所在列的索引或列名
        car_idx: Union[int, str], 车辆id所在列的索引或列名
        x_idx: Union[int, str], 横坐标所在列的索引或列名
        y_idx: Union[int, str], 纵坐标所在列的索引或列名
        lane_idx: Union[int, str], 车道所在列的索引或列名
        v_idx: Union[int, str], 速度所在列的索引或列名
        save_dir: str, 保存图片的文件夹, 如果为None则保存到path所在文件夹
        max_time: int, 最大时间, 默认为1e10
        ids: Union[int, list], 要画的车辆id
        v_trans: bool, 是否将速度转换为km/h
        v_abs: bool, 是否取速度的绝对值
        legend_mode: str, 图例模式, 可选'lane', 'v'
        separate_plot: bool, 是否分开画每个id的轨迹, 默认为False
        '''
        super().__init__(**kwargs)
        if legend_mode == 'lane' and lane_idx is None:
            raise ValueError('If you want to draw lane, you must specify lane_idx.')
        if legend_mode == 'v' and v_idx is None:
            raise ValueError('If you want to draw v, you must specify v_idx.')
        if legend_mode not in ['lane', 'v']:
            raise ValueError(f'legend_mode must be `lane` or `v`, you set {legend_mode}')
        self.path = path
        self.max_time = max_time
        self.ids = ids
        self.v_abs = v_abs
        self.v_trans = v_trans
        self.separate_plot = separate_plot
        self.legend_mode = legend_mode
        # 创建输出文件夹
        self.save_dir = save_dir or path.split('.')[0] + f'_traj_xy_{legend_mode}'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        # 读取数据
        self.df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
        self.time_idx = time_idx if isinstance(time_idx, str) else self.df.columns[time_idx]
        self.car_idx = car_idx if isinstance(car_idx, str) else self.df.columns[car_idx]
        self.x_idx = x_idx if isinstance(x_idx, str) else self.df.columns[x_idx]
        self.y_idx = y_idx if isinstance(y_idx, str) else self.df.columns[y_idx]
        if lane_idx is not None:
            self.lane_idx = lane_idx if isinstance(lane_idx, str) else self.df.columns[lane_idx]
        if v_idx is not None:
            self.v_idx = v_idx if isinstance(v_idx, str) else self.df.columns[v_idx]
        # 数据预处理
        self.df = self.df[self.df[self.time_idx] <= self.max_time]
        if self.ids is not None:
            self.df = self.df[self.df[self.car_idx].isin(self.ids)]
        self.df[self.v_idx] = self.df[self.v_idx] * (3.6 if self.v_trans else 1)
        self.df[self.v_idx] = self.df[self.v_idx].abs() if self.v_abs else self.df[self.v_idx]
        self.df = self.df.sort_values(by=[self.time_idx, self.car_idx], axis=0, ascending=[True, True])
        self.df = self.df.reset_index(drop=True)
        # 车道颜色映射
        self._init_lane_color_map()

    def run(
            self,
            x_min: int = None, x_max: int = None, x_gap: int = 0, x_offset: int = 0,
            y_min: int = None, y_max: int = None, y_gap: int = 0, y_offset: int = 0,
            x_grid: list = None, x_grid_color: str = 'grey', x_grid_style: str = '--', x_grid_width: float = 0.5,
            y_grid: list = None, y_grid_color: str = 'black', y_grid_style: str = '-', y_grid_width: float = 0.5,
            if_line: bool = True, line_width: float = 1, line_style: str = '-',
            if_scatter:bool = True, marker: str = 'o', markersize: float = 1, marker_alpha: float = 0.5,
            origin: bool = True, origin_marker: str = 'o', origin_size: float = 10, origin_color: str = 'red',
            cmap = cm.jet_r, colorbar_min: int = 0, colorbar_max: int = 120, colorbar_step: int = 20,
            ):
        '''运行画图, 根据是否给每个车辆分开画图和图例模式, 进入不同逻辑

        input
        -----
        x_min, x_max, y_min, y_max: float, x轴和y轴范围
        x_gap, y_gap: float, x轴和y轴范围前后间隔
        x_grid, y_grid: list, 竖线网格线和横线网格线的位置
        x_grid_color, y_grid_color: str, 竖线网格线和横线网格线的颜色
        x_grid_style, y_grid_style: str, 竖线网格线和横线网格线的线型
        x_grid_width, y_grid_width: float, 竖线网格线和横线网格线的宽度
        if_line: bool, 是否画线
        line_width, line_style: float, 线宽和线型
        if_scatter: bool, 是否画散点
        markersize, marker_alpha: float, 散点大小和透明度
        origin: bool, 是否画原点
        origin_size, origin_color, origin_marker: float, 原点大小, 颜色, 标记
        cmap: cmap, 颜色映射, 用于速度的颜色映射
        colorbar_min, colorbar_max, colorbar_step: float, 颜色条最小值和最大值, 步长
        '''
        # 预处理
        if not if_line and not if_scatter:
            raise ValueError('if_line and if_scatter cannot both be False. 要不你画啥呢！')
        self.df[self.x_idx] = self.df[self.x_idx] + x_offset
        self.df[self.y_idx] = self.df[self.y_idx] + y_offset
        # 画图函数包装1
        def reserach_plt_xy_funcs():
            '''将ResearchPlt的基本xy框架函数, 打包成一个函数'''
            plt.xlabel(self.x_idx)
            plt.ylabel(self.y_idx)
            self.one_call_xy_settings(
                x_min=x_min, x_max=x_max, x_gap=x_gap,
                y_min=y_min, y_max=y_max, y_gap=y_gap,
                x_grid=x_grid, x_grid_color=x_grid_color, x_grid_style=x_grid_style, x_grid_width=x_grid_width,
                y_grid=y_grid, y_grid_color=y_grid_color, y_grid_style=y_grid_style, y_grid_width=y_grid_width,
            )
        # 画图函数包装2
        def plot_origin_func():
            '''如开启则画原点'''
            if origin:
                plt.scatter([0], [0], s=origin_size, c=origin_color, marker=origin_marker)
        # 车辆独立画图
        if self.separate_plot:
            if self.legend_mode == 'lane':      # lane为legend
                self._plot_by_lane_separately(
                    plt_xy_func=reserach_plt_xy_funcs, plot_origin_func=plot_origin_func,
                    if_line=if_line, line_width=line_width, line_style=line_style,
                    if_scatter=if_scatter, marker=marker, markersize=markersize, marker_alpha=marker_alpha)
            elif self.legend_mode == 'v':       # v为legend
                self._plot_by_speed_separately(
                    plt_xy_func=reserach_plt_xy_funcs, plot_origin_func=plot_origin_func,
                    if_line=if_line, line_width=line_width, line_style=line_style,
                    if_scatter=if_scatter, marker=marker, markersize=markersize, marker_alpha=marker_alpha,
                    cmap=cmap, colorbar_min=colorbar_min, colorbar_max=colorbar_max, colorbar_step=colorbar_step)
        # 全部车辆画图
        else:
            plt.figure()
            plot_origin_func()
            if self.legend_mode == 'lane':
                self._plot_by_lane(
                    if_line=if_line, line_width=line_width, line_style=line_style,
                    if_scatter=if_scatter, marker=marker, markersize=markersize, marker_alpha=marker_alpha)
            elif self.legend_mode == 'v':
                self._plot_by_speed(
                    if_line=if_line, line_width=line_width, line_style=line_style,
                    if_scatter=if_scatter, marker=marker, markersize=markersize, marker_alpha=marker_alpha,
                    cmap=cmap, colorbar_min=colorbar_min, colorbar_max=colorbar_max, colorbar_step=colorbar_step)
            reserach_plt_xy_funcs()
            save_path = os.path.join(self.save_dir, f'traj_xy_{self.legend_mode}_legend.png')
            plt.savefig(save_path)
            plt.close()
        # 恢复数据
        self.df[self.x_idx] = self.df[self.x_idx] - x_offset
        self.df[self.y_idx] = self.df[self.y_idx] - y_offset

    def _plot_by_lane_separately(
        self,
        plt_xy_func: Callable,
        plot_origin_func: Callable,
        if_line: bool, line_width: float, line_style: str,
        if_scatter: bool, marker: str, markersize: float, marker_alpha: float,
        ):
        '''按车道为每个车辆单独画图'''
        handle = tqdm(self.df.groupby(self.car_idx))
        for car_id, car_traj in handle:
            plt.figure()
            plot_origin_func()
            for lane, lane_data in car_traj.groupby(self.lane_idx):
                if if_line and if_scatter:
                    plt.plot(
                        lane_data[self.x_idx], lane_data[self.y_idx],
                        label=lane, color=self.lane_color_map[lane],
                        linewidth=line_width, linestyle=line_style,
                        marker=marker, markersize=markersize, alpha=marker_alpha)
                elif if_line:
                    plt.plot(
                        lane_data[self.x_idx], lane_data[self.y_idx],
                        label=lane, color=self.lane_color_map[lane],
                        linewidth=line_width, linestyle=line_style)
                elif if_scatter:
                    plt.scatter(
                        lane_data[self.x_idx], lane_data[self.y_idx],
                        label=lane, color=self.lane_color_map[lane],
                        marker=marker, s=markersize, alpha=marker_alpha)
            plt_xy_func()
            self.show_legend_sorted(title=self.lane_idx)
            save_path = os.path.join(self.save_dir, f'car_{car_id}_lane_legend.png')
            plt.savefig(save_path)
            plt.close()

    def _plot_by_speed_separately(
        self,
        plt_xy_func: Callable,
        plot_origin_func: Callable,
        if_line: bool, line_width: float, line_style: str,
        if_scatter: bool, marker: str, markersize: float, marker_alpha: float,
        cmap, colorbar_min: int, colorbar_max: int, colorbar_step: int,
        ):
        '''按速度为每个车辆单独画图'''
        norm_func = plt.Normalize(colorbar_min, colorbar_max)
        handle = tqdm(self.df.groupby(self.car_idx))
        for car_id, car_traj in handle:
            handle.set_description(f"car {car_id}")
            num_points = len(car_traj)
            speeds = car_traj[self.v_idx].values
            colors = cmap(norm_func(speeds))
            plt.figure()
            plot_origin_func()
            if if_line:     # 绘制线段，每段线使用对应速度的颜色
                for i in range(num_points - 1):
                    plt.plot(
                        [car_traj[self.x_idx].iloc[i], car_traj[self.x_idx].iloc[i+1]],
                        [car_traj[self.y_idx].iloc[i], car_traj[self.y_idx].iloc[i+1]],
                        color=colors[i], linewidth=line_width, linestyle=line_style)
            if if_scatter:      # 绘制散点
                plt.scatter(
                    car_traj[self.x_idx], car_traj[self.y_idx],
                    c=speeds, cmap=cmap, norm=norm_func,
                    s=markersize, marker=marker, alpha=marker_alpha, zorder=10)
            plt_xy_func()
            self.show_colorbar_speed(
                v_min=colorbar_min, v_max=colorbar_max, v_step=colorbar_step,
                cmap=cmap, label = 'speed (km/h)' if self.v_trans else 'speed (m/s)')
            save_path = os.path.join(self.save_dir, f'car_{car_id}_v_legend.png')
            plt.savefig(save_path)
            plt.close()

    def _plot_by_lane(
        self,
        if_line: bool, line_width: float, line_style: str,
        if_scatter: bool, marker: str, markersize: float, marker_alpha: float,
        ):
        '''按车道将所有轨迹画图'''
        handle = tqdm(self.df.groupby(self.lane_idx))
        for lane, lane_data in handle:
            handle.set_description(f"lane {lane}")
            if if_line and if_scatter:
                plt.plot(lane_data[self.x_idx], lane_data[self.y_idx],
                        label=lane, color=self.lane_color_map[lane],
                        linewidth=line_width, linestyle=line_style,
                        marker=marker, markersize=markersize, alpha=marker_alpha)
            elif if_line:
                for _, car_traj in lane_data.groupby(self.car_idx):
                    plt.plot(car_traj[self.x_idx], car_traj[self.y_idx],
                            label=lane, color=self.lane_color_map[lane],
                            linewidth=line_width, linestyle=line_style)
            elif if_scatter:
                plt.scatter(lane_data[self.x_idx], lane_data[self.y_idx],
                            label=lane, color=self.lane_color_map[lane],
                            marker=marker, s=markersize, alpha=marker_alpha)
        self.show_legend_sorted(title=self.lane_idx)

    def _plot_by_speed(
        self,
        if_line: bool, line_width: float, line_style: str,
        if_scatter: bool, marker: str, markersize: float, marker_alpha: float,
        cmap, colorbar_min: int, colorbar_max: int, colorbar_step: int,
    ):
        '''按速度将所有轨迹画图'''
        norm_func = plt.Normalize(colorbar_min, colorbar_max)
        handle = tqdm(self.df.groupby(self.car_idx))
        for car_id, car_traj in handle:
            handle.set_description(f"car {car_id}")
            num_points = len(car_traj)
            speeds = car_traj[self.v_idx].values
            colors = cmap(norm_func(speeds))
            if if_line:     # 绘制线段，每段线使用对应速度的颜色
                for i in range(num_points - 1):
                    plt.plot(
                        [car_traj[self.x_idx].iloc[i], car_traj[self.x_idx].iloc[i+1]],
                        [car_traj[self.y_idx].iloc[i], car_traj[self.y_idx].iloc[i+1]],
                        color=colors[i], linewidth=line_width, linestyle=line_style)
            if if_scatter:      # 绘制散点
                plt.scatter(
                    car_traj[self.x_idx], car_traj[self.y_idx],
                    c=speeds, cmap=cmap, norm=norm_func,
                    s=markersize, marker=marker, alpha=marker_alpha, zorder=10)
        self.show_colorbar_speed(
            v_min=colorbar_min, v_max=colorbar_max, v_step=colorbar_step,
            cmap=cmap, label ='speed (km/h)' if self.v_trans else'speed (m/s)')

    def _init_lane_color_map(self, cmap = cm.tab10):
        '''初始化车道颜色映射。
        如不需要对车道指定颜色, 可关闭。
        '''
        unique_lanes = self.df[self.lane_idx].unique()
        self.lane_color_map = {lane: cmap(i) for i, lane in enumerate(unique_lanes)}


def main_raoyue():
    '''绕越高速雷达数据平面图'''
    # field data
    x_idx = 3
    y_idx = 4
    lane_idx = 2
    time_idx = 0
    car_idx = 'id'

    # # 画单个文件
    path = r"D:\myscripts\spill-detection\data\extractedData\2024-3-27-17_byDevice\K78+760_1.csv"
    txyp = TrajXyPlot(path=path, x_idx=x_idx, y_idx=y_idx, lane_idx=lane_idx, time_idx=time_idx, car_idx=car_idx)
    txyp.run()


def main_sumo(csv_path: str):
    '''运行sumo仿真的model0轨迹数据'''
    lane_idx = 'laneID'
    car_idx = 'vehicleID'
    time_idx = 'time(s)'
    x_idx= 'x(m)'
    y_idx = 'y(m)'
    v_idx = 'speed(m/s)'
    txyp = TrajXyPlot(
        path=csv_path,
        x_idx=x_idx, y_idx=y_idx, lane_idx=lane_idx,
        car_idx=car_idx, time_idx=time_idx, v_idx=v_idx,
        v_trans=True,
        separate_plot=True, legend_mode='v')
    txyp.run()


if __name__ == '__main__':
    # main_raoyue()
    main_sumo()
