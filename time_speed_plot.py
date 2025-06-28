import os
from typing import Union
import pandas as pd
from tqdm import tqdm
from matplotlib import cm
from matplotlib import pyplot as plt

from .research_plt import ResearchPlt


class TimeSpeedPlotter(ResearchPlt):
    '''时空轨迹图, 继承自ResearchPlt类

    画出指定车辆的时间-速度图像. 横坐标为t, 纵坐标为v
    # TODO 根据example更新代码
    '''
    def __init__(
            self,
            data_path: str,
            time_idx: Union[int, str],
            car_idx: Union[int, str],
            v_idx: Union[int, str],
            save_dir: str = None,
            max_time: int = 1e10,
            ids: Union[int, str, list] = None,
            v_trans: bool = False,
            v_abs: bool = False,
            **kwargs,
            ):
        '''读取并预处理数据

        input
        -----
        data_path: str, 仿真数据csv文件路径
        time_idx: Union[int, str], 帧号列索引或列名
        car_idx: Union[int, str], 车辆ID列索引或列名
        v_idx: Union[int, str], 车辆速度列索引或列名
        save_dir: str, 保存图片的文件夹
        max_time: int, 最大画图的帧数/秒数, 默认值为尽可能大的数字, 即画图范围不限
        ids: Union[int, str, list], 要画图的车辆ID列表, 默认为None, 即所有车辆
        v_trans: bool, 是否转换速度单位, 默认False, 即速度单位为m/s, 设为True则转换为km/h
        v_abs: bool, 是否取速度绝对值, 默认False
        **kwargs: ResearchPlt的初始化参数, 参见ResearchPlt
        '''
        super().__init__(**kwargs)
        self.path = data_path
        self.max_time = max_time
        self.ids = [ids] if isinstance(ids, int) else ids
        self.v_trans = v_trans
        self.v_abs = v_abs
        # 创建输出文件夹
        save_dir = save_dir or os.path.join(os.path.dirname(data_path), 'example')     # NOTE 修改输出文件夹名称
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        # 读取数据
        self.df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_excel(data_path)
        self.time_idx = time_idx if isinstance(time_idx, str) else self.df.columns[time_idx]
        self.car_idx = car_idx if isinstance(car_idx, str) else self.df.columns[car_idx]
        self.v_idx = v_idx if isinstance(v_idx, str) else self.df.columns[v_idx]
        # 数据预处理
        self.df = self.df[self.df[self.time_idx] <= self.max_time]
        if self.ids is not None:
            self.df = self.df[self.df[self.car_idx].isin(self.ids)]
        self.df[self.v_idx] = self.df[self.v_idx] * (3.6 if self.v_trans else 1)
        self.df[self.v_idx] = self.df[self.v_idx].abs() if self.v_abs else self.df[self.v_idx]
        self.df = self.df.sort_values(by=[self.time_idx, self.car_idx], axis=0, ascending=[True, True])
        self.df = self.df.reset_index(drop=True)

    def run(self,
            x_min: int = None, x_max: int = None, x_gap: int = 0, x_offset: int = 0,
            y_min: int = None, y_max: int = None, y_gap: int = 0, y_offset: int = 0,
            x_grid: list = None, x_grid_color: str = 'grey', x_grid_style: str = '--', x_grid_width: float = 0.5,
            y_grid: list = None, y_grid_color: str = 'black', y_grid_style: str = '-', y_grid_width: float = 0.5,
            if_line: bool = False, line_width: float = 1, line_style: str = '-',
            if_scatter: bool = True, marker: str = 'o', markersize: float = 6,
            cmap = cm.jet_r, colorbar_min: int = 0, colorbar_max: int = 120, colorbar_step: int = 20,
            ):
        '''根据初始化的参数画出时空轨迹图

        input
        -----
        x_min, x_max: float, x轴范围
        x_gap: float, x轴范围前后间隔
        y_min, y_max: float, y轴范围
        y_gap: float, y轴范围前后间隔
        x_grid: list, 竖线网格线的x位置
        y_grid: list, 横线网格线的y位置
        x_grid_color, y_grid_color: str, 网格线颜色
        x_grid_style, y_grid_style: str, 网格线样式
        x_grid_width, y_grid_width: float, 网格线宽度
        if_line: bool, 是否画点之间的连接线, 默认为True
        line_width, line_style: float, str, 线的宽度和样式
        if_scatter: bool, 是否画点, 默认为True
        marker, markersize, marker_alpha: str, float, float, 点的样式、大小和透明度
        cmap: cmap, 颜色映射
        colorbar_min, colorbar_max, colorbar_step: int, 颜色条参数
        '''
        # prepare
        if not if_line and not if_scatter:
            raise ValueError('if_line and if_scatter cannot both be False. 要不你画啥呢！')
        self.df[self.time_idx] = self.df[self.time_idx] + x_offset
        self.df[self.v_idx] = self.df[self.v_idx] + y_offset
        # 创建颜色映射
        norm_func = plt.Normalize(colorbar_min, colorbar_max)
        # 画图
        print("begin drawing!")
        handle = tqdm(self.df.groupby(self.car_idx))
        for car_id, car_traj in handle:
            handle.set_description(f"car_id {car_id}")
            num_points = len(car_traj)
            plt.figure()
            # 根据速度值获取颜色
            speeds = car_traj[self.v_idx].values
            colors = cmap(norm_func(speeds))
            # 绘制线段，每段线使用对应速度的颜色
            if if_line:
                for i in range(num_points - 1):
                    plt.plot(
                        [car_traj[self.time_idx].iloc[i], car_traj[self.time_idx].iloc[i+1]],
                        [car_traj[self.v_idx].iloc[i], car_traj[self.v_idx].iloc[i+1]],
                        color=colors[i], linewidth=line_width, linestyle=line_style)
            # 绘制散点
            if if_scatter:
                plt.scatter(
                    car_traj[self.time_idx], car_traj[self.v_idx],
                    c=speeds, cmap=cmap, norm=norm_func,
                    s=markersize, marker=marker, zorder=10)
            # 添加颜色条
            self.show_colorbar_speed(
                cmap=cmap, v_min=colorbar_min, v_max=colorbar_max, v_step=colorbar_step,
                label='speed (km/h)' if self.v_trans else 'speed (m/s)')

            plt.title(f"car_{car_id} time-speed figure")
            plt.xlabel(self.time_idx)
            plt.ylabel('speed (km/h)' if self.v_trans else 'speed (m/s)')
            self.one_call_xy_settings(
                x_min=x_min, x_max=x_max, x_gap=x_gap,
                y_min=y_min, y_max=y_max, y_gap=y_gap,
                x_grid=x_grid, x_grid_color=x_grid_color, x_grid_style=x_grid_style, x_grid_width=x_grid_width,
                y_grid=y_grid, y_grid_color=y_grid_color, y_grid_style=y_grid_style, y_grid_width=y_grid_width,
            )
            plt.savefig(os.path.join(self.save_dir, f"car_{car_id}.jpg"))
            plt.close()
        print("finish drawing!")


def main_sumo_model0():
    '''sumo仿真model0数据画图'''
    path = r'D:\myscripts\pro\output\model0\trajectory.csv'
    # 创建输出文件夹
    save_dir = os.path.join(os.path.dirname(path), 'time_speed')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 数据表列索引
    car_idx = 'vehicleID'
    time_idx = 'time(s)'
    v_idx = 'speed(m/s)'
    # 运行
    tsp = TimeSpeedPlotter(
        data_path=path, save_dir=save_dir,
        car_idx=car_idx, time_idx=time_idx, v_idx=v_idx,
        v_trans=True,
        )
    tsp.run(
        y_min=0, y_max=160, y_gap=0, y_offset=0,
    )

if __name__ == '__main__':
    main_sumo_model0()
