import os
from typing import Union
import pandas as pd
from tqdm import tqdm
from matplotlib import cm
from matplotlib import pyplot as plt


# from .utils import combine_images           # 考虑到需要被外部代码引用，因此一定需要相对引入, 否则外部引入后这些对应的代码将会从外部代码导入，而非本文件夹所构建
# from .research_plt import ResearchPlt
from utils import combine_images           # 考虑到需要被外部代码引用，因此一定需要相对引入, 否则外部引入后这些对应的代码将会从外部代码导入，而非本文件夹所构建
from research_plt import ResearchPlt

class TimeSpacePlotter(ResearchPlt):
    '''时空轨迹图, 继承自ResearchPlt类
    
    时空轨迹图的横坐标为帧号/时间, 纵坐标为车辆(沿行驶方向)位置, 图中颜色表示车辆速度。
    时空轨迹图将按lane画图, 每个lane的车辆轨迹将分别画在同一张图中。
    '''
    def __init__(
            self,
            
            time_idx: Union[int, str],
            car_idx: Union[int, str],
            lane_idx: Union[int, str],
            dist_idx: Union[int, str],
            v_idx: Union[int, str, list],
            path: str = None,
            df: pd.DataFrame = None,
            output_dir: str = None,
            max_time: int = 1e10,
            ids: Union[list, str, int] = None,
            v_trans: bool = False,
            v_abs: bool = False,
            open_click: bool = True,

            **kwargs,
            ):
        '''读取并预处理数据

        input
        -----
        path: str, 仿真数据csv文件路径
        df: pd.DataFrame, 仿真数据DataFrame, 默认为None, 若提供则path参数将被忽略
        time_idx: Union[int, str], 帧号列索引或列名
        car_idx: Union[int, str], 车辆ID列索引或列名
        lane_idx: Union[int, str], 车道号列索引或列名
        dist_idx: Union[int, str], 车辆位置列索引或列名.可设为auto，将自动从x或y列选择，选取范围更大的那一个
        v_idx: Union[int, str], 车辆速度列索引或列名。可设为速度分量列的列表，将计算合速度列speed
        output_dir: str, 保存图片的文件夹
        max_time: int, 最大画图的帧数/秒数, 默认值为尽可能大的数字, 即画图范围不限
        ids: Union[int, str, list], 要画图的车辆ID列表, 默认为None, 即所有车辆
        v_trans: bool, 是否转换速度单位, 默认False, 即速度单位为m/s, 设为True则转换为km/h
        v_abs: bool, 是否取速度绝对值, 默认False
        open_click: 支持点击显示点信息
        **kwargs: ResearchPlt的初始化参数, 参见ResearchPlt
        '''
        # path和df必须且能提供一个
        if (path is None) == (df is None):
            raise ValueError('path和df必须且能提供一个')
        super().__init__(**kwargs)
        self.path = path
        self.output_dir = output_dir
        self.max_time = max_time
        self.ids = [ids] if isinstance(ids, int) else ids
        self.v_trans = v_trans
        self.v_abs = v_abs
        self.open_click = open_click

        # 读取数据
        if df is None:
            self.df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
        else:
            self.df = df
        self.lane_idx = lane_idx if isinstance(lane_idx, str) else self.df.columns[lane_idx]
        self.car_idx = car_idx if isinstance(car_idx, str) else self.df.columns[car_idx]
        self.time_idx = time_idx if isinstance(time_idx, str) else self.df.columns[time_idx]
        if dist_idx == 'auto':
            self.dist_idx = 'x' if self.df['x'].max() - self.df['x'].min() > self.df['y'].max() - self.df['y'].min() else 'y'
        else:
            self.dist_idx = dist_idx if isinstance(dist_idx, str) else self.df.columns[dist_idx]
        if isinstance(v_idx, list):
            vx_idx, vy_idx = v_idx
            self.v_idx = 'speed'
            self.df[self.v_idx] = (self.df[vx_idx]**2 + self.df[vy_idx]**2)**0.5
        else:
            self.v_idx = v_idx if isinstance(v_idx, str) else self.df.columns[v_idx]
        self.df = self.df.sort_values(by=[self.time_idx, self.car_idx, self.lane_idx],
                                          axis=0, ascending=[True, True, True])
        # 数据预处理
        self.df = self.df[self.df[self.time_idx] <= self.max_time]
        if self.ids is not None:
            self.df = self.df[self.df[self.car_idx].isin(self.ids)]
        self.df[self.v_idx] = self.df[self.v_idx] * (3.6 if self.v_trans else 1)
        self.df[self.v_idx] = self.df[self.v_idx].abs() if self.v_abs else self.df[self.v_idx]
        self.df = self.df.sort_values(by=[self.time_idx, self.car_idx], axis=0, ascending=[True, True])
        self.df = self.df.reset_index(drop=True)

    def run(self,
            marker: str = 'o', markersize: float = 1, marker_alpha: float = 0.5, combine: bool = True,
            x_min: int = None, x_max: int = None, x_gap: int = 0, x_offset: int = 0,
            y_min: int = None, y_max: int = None, y_gap: int = 0, y_offset: int = 0,
            x_grid: list = None, x_grid_color: str = 'grey', x_grid_style: str = '--', x_grid_width: float = 0.5,
            y_grid: list = None, y_grid_color: str = 'black', y_grid_style: str = '-', y_grid_width: float = 0.5,
            cmap = cm.jet_r, colorbar_min: int = 0, colorbar_max: int = 120, colorbar_step: int = 20,
            name_label: str = '', xlabel: str = None, ylabel: str = None,
            ):
        '''根据初始化的参数画出时空轨迹图

        input
        -----
        marker: str, 点的形状
        markersize: float, 点的大小
        marker_alpha: float, 点的透明度
        x_min, x_max: float, x轴范围
        x_gap: float, x轴范围前后间隔
        y_min, y_max: float, y轴范围
        y_gap: float, y轴范围前后间隔
        x_grid: list, 竖线网格线的x位置
        y_grid: list, 横线网格线的y位置
        x_grid_color, y_grid_color: str, 网格线颜色
        x_grid_style, y_grid_style: str, 网格线样式
        x_grid_width, y_grid_width: float, 网格线宽度
        cmap: str, 速度颜色映射, 默认为'jet_r'
        colorbar_min, colorbar_max, colorbar_step: int, 颜色条参数
        name_label: str, 保存图片时的文件名前缀, 默认为空字符串
        '''
        # 数据预处理
        self.df[self.time_idx] = self.df[self.time_idx] + x_offset
        self.df[self.dist_idx] = self.df[self.dist_idx] + y_offset
        norm_func = plt.Normalize(colorbar_min, colorbar_max)
        if x_max is None:
            x_max = self.df[self.time_idx].max()
        # 画图
        print("begin drawing!")
        handle = tqdm(self.df.groupby(self.lane_idx))
        for lane, lane_data in handle:
            # one figure for each lane
            handle.set_description(f"lane {lane}")
            # plt.figure()
            fig, ax = plt.subplots()
            scatter_objs = []
            vehicle_ids = []
            for car_id, car_traj in lane_data.groupby(lane_data[self.car_idx]):
                car_traj[self.v_idx] = car_traj[self.v_idx]
                speeds = car_traj[self.v_idx].values
                sc = plt.scatter(car_traj[self.time_idx], car_traj[self.dist_idx],
                            c=speeds, cmap=cmap, norm=norm_func,
                            marker=marker, s=markersize, alpha=marker_alpha)
                scatter_objs.append(sc)
                vehicle_ids.append(car_id)
            if self.open_click:
                self.setup_click_handler(fig, ax, scatter_objs, vehicle_ids)

            plt.title(f"lane {lane}")
            self.show_colorbar_speed(
                cmap=cmap, v_min=colorbar_min, v_max=colorbar_max, v_step=colorbar_step)
            plt.xlabel(xlabel or self.time_idx)
            plt.ylabel(ylabel or self.dist_idx)
            self.one_call_xy_settings(
                x_min=x_min, x_max=x_max, x_gap=x_gap,
                y_min=y_min, y_max=y_max or lane_data[self.dist_idx].max(), y_gap=y_gap,
                x_grid=x_grid, x_grid_color=x_grid_color, x_grid_style=x_grid_style, x_grid_width=x_grid_width,
                y_grid=y_grid, y_grid_color=y_grid_color, y_grid_style=y_grid_style, y_grid_width=y_grid_width,
            )
            plt.grid()
            plt.savefig(os.path.join(self.output_dir, f"ts-{name_label}_lane_{lane}.jpg"))
            # if lane >= 1:     # 指定情况，通过此处设置显示lane
            #     plt.show()
            plt.close()
        if combine:
            combine_images(self.output_dir, pattern=f"ts-{name_label}_lane_", output_name=f'ts-{name_label}-combined.jpg')
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
        v_trans=True, markersize=1,
        figsize=(20,12),
        )
    tsp.run()


def main_sumo_model0(output_dir: str):
    '''sumo仿真model0数据画图
    
    input
    -----
    output_dir: str, 仿真数据文件夹路径
    '''
    # path = r'D:\myscripts\pro\output\model0\trajectory.csv'
    path = os.path.join(output_dir, 'trajectory.csv')
    # 创建输出文件夹
    output_dir = output_dir or os.path.join(os.path.dirname(path), 'trajectory')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 数据表列索引
    lane_idx = 'laneID'
    car_idx = 'vehicleID'
    time_idx = 'time(s)'
    # dist_idx = 'odeometer(m)'
    dist_idx= 'x(m)'
    v_idx = 'speed(m/s)'
    # 运行
    tsp = TimeSpacePlotter(
        path=path, output_dir=output_dir,
        lane_idx=lane_idx, car_idx=car_idx, time_idx=time_idx, dist_idx=dist_idx, v_idx=v_idx,
        v_trans=True, v_abs=True,
        figsize=(20,8),
        )
    tsp.run(
        markersize=0.5, marker_alpha=0.25,
        x_min=0, x_max=1600,
        y_min=0, y_max=2500, y_gap=50, y_offset=1000,
        y_grid=[1000, 1500],
        )


def main_sumo_model1(output_dir: str):
    '''sumo仿真model1数据画图
    
    input
    -----
    output_dir: str, 仿真数据文件夹路径
    '''
    # path = r'D:\myscripts\pro\output\model1\trajectory.csv'
    path = os.path.join(output_dir, 'trajectory.csv')
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 数据表列索引
    lane_idx = 'laneID'
    car_idx = 'vehicleID'
    time_idx = 'time(s)'
    # dist_idx = 'odeometer(m)'
    dist_idx= 'x(m)'
    v_idx = 'speed(m/s)'
    # 运行
    tsp = TimeSpacePlotter(
        path=path, output_dir=output_dir,
        lane_idx=lane_idx, car_idx=car_idx, time_idx=time_idx, dist_idx=dist_idx, v_idx=v_idx,
        v_trans=True, v_abs=True,
        figsize=(10,8),
        )
    tsp.run(
        markersize=0.5, marker_alpha=1,
        x_min=0, x_max=1400,
        y_min=0, y_max=1700, y_gap=0,       # y_max=1700
        y_offset=1000,
        y_grid=[300, 1000, 1200],
        cmap=cm.jet_r,
        )

def main_sumo_model2(output_dir: str):
    '''sumo仿真model2数据画图

    input
    -----
    output_dir: str, 仿真数据文件夹路径
    '''
    # path = r'D:\myscripts\pro\output\model2\trajectory.csv'
    path = os.path.join(output_dir, 'trajectory.csv')
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 数据表列索引
    lane_idx = 'laneID'
    car_idx = 'vehicleID'
    time_idx = 'time(s)'
    # dist_idx = 'odeometer(m)'
    dist_idx= 'x(m)'
    v_idx = 'speed(m/s)'
    # 运行
    tsp = TimeSpacePlotter(
        path=path, output_dir=output_dir,
        lane_idx=lane_idx, car_idx=car_idx, time_idx=time_idx, dist_idx=dist_idx, v_idx=v_idx,
        v_trans=True, v_abs=True,
        figsize=(10,8),
        )
    tsp.run(
        markersize=0.5, marker_alpha=1,
        x_min=0, x_max=1400,
        y_min=0, y_max=1700, y_gap=0,
        y_offset=1000,
        y_grid=[300, 800, 1000, 1200],
        cmap=cm.jet_r,
        )


def main_sumo_single_car(output_dir: str):
    '''sumo仿真数据画图, 单车模型

    input
    -----
    output_dir: str, 仿真数据文件夹路径
    '''
    # path = r'D:\myscripts\pro\output\single_car\trajectory.csv'
    path = os.path.join(output_dir, 'trajectory.csv')
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 数据表列索引
    lane_idx = 'laneID'
    car_idx = 'vehicleID'
    time_idx = 'time(s)'
    # dist_idx = 'odeometer(m)'
    dist_idx= 'x(m)'
    v_idx = 'speed(m/s)'
    # 运行
    tsp = TimeSpacePlotter(
        path=path, output_dir=output_dir,
        lane_idx=lane_idx, car_idx=car_idx, time_idx=time_idx, dist_idx=dist_idx, v_idx=v_idx,
        v_trans=True, v_abs=True,
        figsize=(10,8),
        )
    tsp.run(
        combine=False,
        markersize=0.5, marker_alpha=1,
        x_min=0, x_max=80,
        y_min=0, y_max=600,
        y_grid=[200, 400],
        # cmap=cm.jet_r,
        )


def main_sumo_model3(output_dir: str):
    '''sumo仿真model3数据画图
    
    input
    -----
    output_dir: str, 仿真数据文件夹路径
    '''
    # path = r'D:\myscripts\pro\output\model1\trajectory.csv'
    path = os.path.join(output_dir, 'trajectory.csv')
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 数据表列索引
    lane_idx = 'laneID'
    car_idx = 'vehicleID'
    time_idx = 'time(s)'
    # dist_idx = 'odeometer(m)'
    dist_idx= 'x(m)'
    v_idx = 'speed(m/s)'
    # 运行
    tsp = TimeSpacePlotter(
        path=path, output_dir=output_dir,
        lane_idx=lane_idx, car_idx=car_idx, time_idx=time_idx, dist_idx=dist_idx, v_idx=v_idx,
        v_trans=True, v_abs=True,
        figsize=(14,8),
        )
    tsp.run(
        markersize=0.5, marker_alpha=1,
        x_min=0, x_max=1400,     # 500
        y_min=0, y_max=1700, y_gap=0,
        y_offset=0,     # 300, 现在把预热的300m去掉，不画在图上
        # y_grid=[300, 800, 1300, 1500],
        y_grid=[500, 1000, 1200],
        cmap=cm.jet_r,
        )


def main_sumo_model4(output_dir: str):
    '''sumo仿真model3数据画图
    
    input
    -----
    output_dir: str, 仿真数据文件夹路径
    '''
    # path = r'D:\myscripts\pro\output\model1\trajectory.csv'
    path = os.path.join(output_dir, 'trajectory.csv')
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 数据表列索引
    lane_idx = 'laneID'
    car_idx = 'vehicleID'
    time_idx = 'time(s)'
    # dist_idx = 'odeometer(m)'
    dist_idx= 'x(m)'
    v_idx = 'speed(m/s)'
    # 运行
    tsp = TimeSpacePlotter(
        path=path, output_dir=output_dir,
        lane_idx=lane_idx, car_idx=car_idx, time_idx=time_idx, dist_idx=dist_idx, v_idx=v_idx,
        v_trans=True, v_abs=True,
        figsize=(14,8),
        )
    tsp.run(
        markersize=0.5, marker_alpha=1,
        x_min=0, x_max=1600,     # 500
        y_min=0, y_max=3700, y_gap=0,
        y_offset=0,     # 300, 现在把预热的300m去掉，不画在图上
        # y_grid=[300, 800, 1300, 1500],
        y_grid=[1500, 2865, 3000, 3200],
        cmap=cm.jet_r,
        )


def main_sumo_model5(output_dir: str):
    '''sumo仿真model5数据画图
    
    input
    -----
    output_dir: str, 仿真数据文件夹路径
    '''
    # path = r'D:\myscripts\pro\output\model1\trajectory.csv'
    path = os.path.join(output_dir, 'trajectory.csv')
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 数据表列索引
    lane_idx = 'laneID'
    car_idx = 'vehicleID'
    time_idx = 'time(s)'
    # dist_idx = 'odeometer(m)'
    dist_idx= 'x(m)'
    v_idx = 'speed(m/s)'
    # 运行
    tsp = TimeSpacePlotter(
        path=path, output_dir=output_dir,
        lane_idx=lane_idx, car_idx=car_idx, time_idx=time_idx, dist_idx=dist_idx, v_idx=v_idx,
        v_trans=True, v_abs=True, open_click=True,
        figsize=(14,8),
        )
    tsp.run(
        markersize=0.5, marker_alpha=1,
        x_min=0, x_max=1200,     # 500
        y_min=0, y_max=3000, y_gap=0,
        y_offset=0,     # 300, 现在把预热的300m去掉，不画在图上
        # y_grid=[300, 800, 1300, 1500],
        y_grid=[300, 1300, 2165, 2300, 2500, 3000],
        cmap=cm.jet_r,
        )


def main_default(output_dir: str):
    '''sumo仿真画图, 未指定模型时的默认调用函数

    input
    -----
    output_dir: str, 仿真数据文件夹路径
    '''
    # file
    path = os.path.join(output_dir, 'trajectory.csv')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # data
    lane_idx = 'laneID'
    car_idx = 'vehicleID'
    time_idx = 'time(s)'
    # dist_idx = 'odeometer(m)'
    dist_idx= 'x(m)'
    v_idx = 'speed(m/s)'
    # plot
    tsp = TimeSpacePlotter(
        path=path, output_dir=output_dir,
        lane_idx=lane_idx, car_idx=car_idx, time_idx=time_idx, dist_idx=dist_idx, v_idx=v_idx,
        v_trans=True, v_abs=True,
        # figsize=(16,8),
        figsize=(16,8),
        )
    tsp.run(
        markersize=0.5, marker_alpha=1,
        x_min=0, y_min=0,
        cmap=cm.jet_r,
        )


def time_space_plot_by_sumo_model(model: str, output_dir: str):
    '''根据传入的model名称, 调用不同函数(针对各个模型设置了不同参数)进行画图

    input
    -----
    model: str, 模型名称, 可选model0, model1, model2,...
    '''
    model = model.split('-')[0]     # model3-1, model3-2, model3-3,...
    func_map = {
        'model0': main_sumo_model0,
        'model1': main_sumo_model1,
        'model2': main_sumo_model2,
        'model3': main_sumo_model3,
        'model4': main_sumo_model4,
        'model5': main_sumo_model5,
        'single_car': main_sumo_single_car,
    }
    if model not in func_map:
        print(f"WARNING: model {model} not in {func_map.keys()}, use default function")
        main_default(output_dir)
        return
    func_map[model](output_dir)


def main_ute(path: str):
    '''ute数据集的数据画图'''
    output_dir = path.removesuffix('.csv')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 数据表列索引
    lane_idx = 'laneID'
    car_idx = 'vehicleID'
    time_idx = 'time(s)'
    dist_idx = 'longitudinalDistance(m)'
    v_idx = 'speed(km/h)'
    # 运行
    tsp = TimeSpacePlotter(
        path=path, output_dir=output_dir,
        lane_idx=lane_idx, car_idx=car_idx, time_idx=time_idx,
        dist_idx=dist_idx, v_idx=v_idx,
        v_trans=False, v_abs=True, open_click=False,
        figsize=(20,10),
        )
    tsp.run(markersize=1,x_min=0)


def main_datasets(
    parquet_dir: str = r'E:\datasets\a微观轨迹数据集\parquet_original_value\train',
    datasets: list = ['highD', 'Mitra', 'NGSIM', 'ZEN', 'RAOYUE'],
    ):
    '''从parquet格式快速读取数据集并画图'''
    from parquet_read import ParquetReader
    parquet_reader = ParquetReader(parquet_dir)
    for dataset in datasets:
        file_ids = parquet_reader.getFileIds(dataset)
        for file_id in file_ids:
            df = parquet_reader.load_dataset(
                dataset_name=dataset, file_id=file_id,
                )
            print(f'readed: {dataset}-{file_id}, shape: {df.shape}')
            tsp = TimeSpacePlotter(
                df=df, output_dir=r'E:\datasets\a微观轨迹数据集\time_space_plot',
                lane_idx='lane', car_idx='id', time_idx='frame',
                dist_idx='auto', v_idx=['vx', 'vy'],
                v_trans=True, v_abs=True, open_click=False,
                figsize=(20,10),
                )
            tsp.run(markersize=1,x_min=0, name_label=f"{dataset}-{file_id}")


def main_dataset(path: str):
    '''标准数据格式画图'''
    output_dir = path.removesuffix('.csv')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 数据表列索引
    lane_idx = 'lane'
    car_idx = 'id'
    time_idx = 'frame'
    dist_idx = 'curve_distance'
    v_idx = ['vx', 'vy']
    # 运行
    tsp = TimeSpacePlotter(
        path=path, output_dir=output_dir,
        lane_idx=lane_idx, car_idx=car_idx, time_idx=time_idx,
        dist_idx=dist_idx, v_idx=v_idx,
        v_trans=True, v_abs=True, open_click=False,
        figsize=(24,8),
        font_size=30,
        # figsize=(20,8),
        # font_size=30,
        )
    tsp.run(markersize=1,x_min=0, y_min=0,
        xlabel='帧', ylabel='距离 (m)',
    )


if __name__ == '__main__':
    # main_example()
    # main_sumo_model0()
    # DIR = r'D:\myscripts\pro\output\model1_20250604-134851-下游1km'
    # DIR = r'D:\myscripts\pro\output\model1_20250604-140443-下游2km'
    # DIR = r'D:\myscripts\pro\output\model1_20250604-220051-下游3km'
    # main_sumo_model1(DIR)
    # main_sumo_model2()
    # main_sumo_single_car()
    # test_sumo_model0_single_car()

    # 画UTE
    # path = r'D:\东南大学\科研\UTE数据集\SQM\SQM-W-1\frenet.csv'
    # main_ute(path)

    # 画多个数据集
    # main_datasets()

    path = r'E:\datasets\a微观轨迹数据集\frenet\Mitra\frenet_Mitra_4.csv'
    path = r'E:\datasets\a微观轨迹数据集\frenet\highD\frenet_highD_12.csv'
    path = r'E:\datasets\a微观轨迹数据集\frenet\NGSIM\frenet_NGSIM_us-101.csv'
    path = r'E:\datasets\a微观轨迹数据集\frenet\ZEN\frenet_ZEN_L001_F005.csv'
    path = r'E:\datasets\a微观轨迹数据集\frenet\RAOYUE\frenet_RAOYUE_K79+886.csv'
    main_dataset(path)
