import os
from typing import Union
import pandas as pd
from tqdm import tqdm
from matplotlib import cm
from matplotlib import pyplot as plt


from research_plt import ResearchPlt


class TimeSpacePlotter(ResearchPlt):
    '''时空轨迹图, 继承自ResearchPlt类
    
    时空轨迹图的横坐标为帧号/时间, 纵坐标为车辆(沿行驶方向)位置, 图中颜色表示车辆速度。
    时空轨迹图将按lane画图, 每个lane的车辆轨迹将分别画在同一张图中。
    '''
    def __init__(
            self,
            path: str,
            output_dir: str,
            lane_idx: Union[int, str],
            car_idx: Union[int, str],
            time_idx: Union[int, str],
            dist_idx: Union[int, str],
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
        path: str, 仿真数据csv文件路径
        output_dir: str, 保存图片的文件夹
        lane_idx: Union[int, str], 车道号列索引或列名
        car_idx: Union[int, str], 车辆ID列索引或列名
        time_idx: Union[int, str], 帧号列索引或列名
        dist_idx: Union[int, str], 车辆位置列索引或列名
        v_idx: Union[int, str], 车辆速度列索引或列名
        max_time: int, 最大画图的帧数/秒数, 默认值为尽可能大的数字, 即画图范围不限
        v_trans: bool, 是否转换速度单位, 默认False, 即速度单位为m/s, 设为True则转换为km/h
        colormap: cm, 画图参数，颜色映射, 默认为逆序的彩虹色rainbow_r
        scatter_size: int, 画图参数，轨迹的散点大小, 默认为1
        **kwargs: ResearchPlt的初始化参数, 参见ResearchPlt
        '''
        super().__init__(**kwargs)
        self.path = path
        self.output_dir = output_dir
        self.max_time = max_time
        self.v_trans = v_trans
        self.colormap = colormap
        self.scatter_size = scatter_size
        # 读取数据
        self.data = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
        self.lane_idx = lane_idx if isinstance(lane_idx, str) else self.data.columns[lane_idx]
        self.car_idx = car_idx if isinstance(car_idx, str) else self.data.columns[car_idx]
        self.time_idx = time_idx if isinstance(time_idx, str) else self.data.columns[time_idx]
        self.dist_idx = dist_idx if isinstance(dist_idx, str) else self.data.columns[dist_idx]
        self.v_idx = v_idx if isinstance(v_idx, str) else self.data.columns[v_idx]
        self.data = self.data.sort_values(by=[self.lane_idx, self.car_idx, self.time_idx],
                                          axis=0, ascending=[True, True, True])
        self.data = self.data.reset_index(drop=True)

    def run(self,
            x_min: int = None, x_max: int = None, x_gap: int = None,
            y_min: int = None, y_max: int = None, y_gap: int = None,
            x_grid: list = None, x_grid_color: str = 'grey',
            x_grid_style: str = '--', x_grid_width: float = 0.5,
            y_grid: list = None, y_grid_color: str = 'black',
            y_grid_style: str = '-', y_grid_width: float = 0.5,
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
        '''
        # 数据预处理
        data = self.data[self.data[self.time_idx] < self.max_time]
        data[self.v_idx] = data[self.v_idx] * 3.6 if self.v_trans else data[self.v_idx]
        # 画图
        print("begin drawing!")
        for lane, lane_data in tqdm(data.groupby(data[self.lane_idx])):
            plt.figure()
            for _, car_traj in lane_data.groupby(lane_data[self.car_idx]):
                car_traj[self.v_idx] = car_traj[self.v_idx].abs()  # 取abs，以免v方向与指定方向相反而为负
                plt.scatter(car_traj[self.time_idx], car_traj[self.dist_idx],
                            c=list(car_traj[self.v_idx]), cmap=self.colormap, s=self.scatter_size)
            plt.title(f"lane {lane} trajectories")
            plt.colorbar()
            plt.xlabel(self.time_idx)
            plt.ylabel(self.dist_idx)
            self.xy_limit_with_gap(
                x_min=x_min, x_max=x_max, x_gap=x_gap,
                y_min=y_min, y_max=y_max, y_gap=y_gap,
                )
            self.specified_grid(
                x_grid=x_grid, x_grid_color=x_grid_color,
                x_grid_style=x_grid_style, x_grid_width=x_grid_width,
                y_grid=y_grid, y_grid_color=y_grid_color,
                y_grid_style=y_grid_style, y_grid_width=y_grid_width,
                )
            plt.savefig(os.path.join(self.output_dir, f"lane_{lane}.jpg"))
            plt.close()
        print("finish drawing!")


def main_example():
    '''样例数据画图'''
    path = 'data/tra_sample.xlsx'
    output_dir = path.removesuffix('.xlsx')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 数据表列索引
    lane_idx = 1
    car_idx = 0
    time_idx = 2
    dist_idx = 3
    v_idx = 6
    # 运行
    tsp = TimeSpacePlotter(
        path=path, output_dir=output_dir,
        lane_idx=lane_idx, car_idx=car_idx, time_idx=time_idx,
        dist_idx=dist_idx, v_idx=v_idx,
        v_trans=True, scatter_size=2,
        figsize=(20,8),
        )
    tsp.run()


def main_sumo_model0():
    '''sumo仿真model0数据画图'''
    path = r'D:\myscripts\pro\output\test0_post.csv'
    # 创建输出文件夹
    output_manager_dir = path.strip('.csv')
    if not os.path.exists(output_manager_dir):
        os.makedirs(output_manager_dir)
    lane_change_output_dir = os.path.join(output_manager_dir, 'trajectory')
    if not os.path.exists(lane_change_output_dir):
        os.makedirs(lane_change_output_dir)
    # 数据表列索引
    lane_idx = 2
    car_idx = 0
    time_idx = 1
    dist_idx = 8
    v_idx = 9
    # 运行
    tsp = TimeSpacePlotter(
        path=path, output_dir=lane_change_output_dir,
        lane_idx=lane_idx, car_idx=car_idx, time_idx=time_idx,
        dist_idx=dist_idx, v_idx=v_idx,
        v_trans=True, scatter_size=2,
        figsize=(20,8),
        )
    tsp.run(
        x_min=0, x_max=2000,
        y_min=0, y_max=2500, y_gap=50,
        y_grid=[1000, 1500],
        )


if __name__ == '__main__':
    main_example()
    # main_sumo_model0()
