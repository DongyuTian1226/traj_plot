# pylint: disable=unused-argument, unused-variable
import os
from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

from research_plt import ResearchPlt


class ExamplePlotter(ResearchPlt):
    '''示例, 基于ResearchPlt类进行继承。
    该class作为新class的代码模板, 使用请检查每一个NOTE提示。

    设计规则
    -------
    将数据读取和画图的代码块分离, 避免代码和运行存在耦合关系, 代码清晰高效。
    初始化: 输入数据文件配置, 数据列配置, 执行数据读取和预处理。
    run: 输入画图的超参数和plt配置, 执行基本的科研风格绘图。

    优点
    -----
    1. 代码清晰, 易于理解, 易于维护
    2. 画图的参数和数据读取的参数分离, 避免代码和运行存在耦合关系
    3. 绘图风格基于科研风格, 易于传播
    4. 显示定义plt常用画图参数, 易于使用

    缺点
    -----
    1. 显示定义plt常用画图参数过多, 源码较长, 易忘记配置
    2. 自定义的程度不如直接使用plt画图灵活
    '''
    def __init__(
        self,
        data_path: str,
        time_idx: Union[int, str],
        car_idx: Union[int, str],
        lane_idx: Union[int, str],
        x_idx: Union[int, str],
        y_idx: Union[int, str],
        dist_idx: Union[int, str],
        v_idx: Union[int, str],
        save_dir: str = None,
        max_time: int = 1e10,
        ids: Union[int, str, list] = None,
        v_trans: bool = False,
        v_abs: bool = False,
        # NOTE: 如不需要部分idx参数, 可删除或设为可选参数
        # NOTE: 在此添加数据相关参数
        **kwargs):
        '''
        input
        -----
        data_path: str, 数据文件路径
        time_idx: Union[int, str], 时间索引
        car_idx: Union[int, str], 车辆ID索引
        lane_idx: Union[int, str], 车道索引
        x_idx: Union[int, str], 二维坐标系下x坐标索引
        y_idx: Union[int, str], 二维坐标系下y坐标索引
        dist_idx: Union[int, str], 车辆位置索引
        v_idx: Union[int, str], 车辆速度索引
        save_dir: str, 保存图片的文件夹
        max_time: int, 最大时间, 默认为1e10, 即所有时间(单位可以为s或帧)
        ids: Union[int, str, list], 要画图的车辆ID列表, 默认为None, 即所有车辆
        v_trans: bool, 是否转换速度单位, 默认False, 即速度单位为m/s, 设为True则转换为km/h
        v_abs: bool, 是否取速度绝对值, 默认False
        **kwargs: ResearchPlt的初始化画图参数, 图像大小、字体大小等, 详见ResearchPlt
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
        self.lane_idx = lane_idx if isinstance(lane_idx, str) else self.df.columns[lane_idx]
        self.x_idx = x_idx if isinstance(x_idx, str) else self.df.columns[x_idx]
        self.y_idx = y_idx if isinstance(y_idx, str) else self.df.columns[y_idx]
        self.dist_idx = dist_idx if isinstance(dist_idx, str) else self.df.columns[dist_idx]
        self.v_idx = v_idx if isinstance(v_idx, str) else self.df.columns[v_idx]
        # 数据预处理
        self.df = self.df[self.df[self.time_idx] <= self.max_time]
        if self.ids is not None:
            self.df = self.df[self.df[self.car_idx].isin(self.ids)]
        self.df[self.v_idx] = self.df[self.v_idx] * (3.6 if self.v_trans else 1)
        self.df[self.v_idx] = self.df[self.v_idx].abs() if self.v_abs else self.df[self.v_idx]
        self.df = self.df.sort_values(by=[self.time_idx, self.car_idx], axis=0, ascending=[True, True])
        self.df = self.df.reset_index(drop=True)
        self._init_lane_color_map()     # NOTE: 不需要lane颜色时可注释

    def run(self,
            x_min: int = None, x_max: int = None, x_gap: int = 0, x_offset: int = 0,
            y_min: int = None, y_max: int = None, y_gap: int = 0, y_offset: int = 0,
            x_grid: list = None, x_grid_color: str = 'grey', x_grid_style: str = '--', x_grid_width: float = 0.5,
            y_grid: list = None, y_grid_color: str = 'black', y_grid_style: str = '-', y_grid_width: float = 0.5,
            if_line: bool = False, line_width: float = 1, line_style: str = '-', line_alpha: float = 1,
            if_scatter:bool = False, marker: str = 'o', markersize: float = 1, marker_alpha: float = 1,
            cmap = cm.jet_r, colorbar_min: int = 0, colorbar_max: int = 120, colorbar_step: int = 20,
            # NOTE: 在此添加画图相关参数
            # NOTE: 调整默认if_line和if_scatter
            ):
        '''
        input
        -----
        x_min, x_max, y_min, y_max: float, x轴和y轴范围
        x_gap, y_gap: float, x轴和y轴两侧预留的间距
        x_offset, y_offset: float, x轴和y轴数据的偏移量
        x_grid, y_grid: list, x轴和y轴的网格线位置
        x_grid_color, y_grid_color: str, x轴和y轴网格线的颜色
        x_grid_style, y_grid_style: str, x轴和y轴网格线的样式
        x_grid_width, y_grid_width: float, x轴和y轴网格线的宽度
        if_line: bool, 是否画点之间的连接线, 默认为True
        line_width, line_style: float, str, 线的宽度和样式
        if_scatter: bool, 是否画点, 默认为True
        marker, markersize, marker_alpha: str, float, float, 点的样式、大小和透明度
        cmap: str, 速度颜色映射, 默认为'jet_r'
        colorbar_min, colorbar_max, colorbar_step: int, int, int, 颜色条的最小值、最大值和步长
        '''
        # prepare
        if not if_line and not if_scatter:
            raise ValueError('if_line and if_scatter cannot both be False. 要不你画啥呢！')
        self.df[self.x_idx] = self.df[self.x_idx] + x_offset
        self.df[self.y_idx] = self.df[self.y_idx] + y_offset
        plt.figure()    # NOTE: 如做多图, 需移动到for循环内
        # plt
        print("begin drawing!")
        # NOTE: 在此添加你的画图代码, 下为示例
        handle = tqdm(self.df.groupby(self.df[self.lane_idx]))    # NOTE: 以分车道为例
        for lane, lane_data in handle:
            handle.set_description(f"drawing lane {lane}")
        # plt element: xy
        self.one_call_xy_settings(
            x_min=x_min, x_max=x_max, x_gap=x_gap,
            y_min=y_min, y_max=y_max, y_gap=y_gap,
            x_grid=x_grid, x_grid_color=x_grid_color, x_grid_style=x_grid_style, x_grid_width=x_grid_width,
            y_grid=y_grid, y_grid_color=y_grid_color, y_grid_style=y_grid_style, y_grid_width=y_grid_width,
        )
        # plt element: legend/colorbar
        # NOTE: 在此修改或开启legend或colorbar, 一般二选一开一个
        self.show_legend_sorted(title=None)
        self.show_colorbar_speed(cmap=cmap, v_min=colorbar_min, v_max=colorbar_max, v_step=colorbar_step,
                                 label='speed(km/h)' if self.v_trans else 'speed(m/s)')

        # end: restore data
        self.df[self.x_idx] = self.df[self.x_idx] - x_offset
        self.df[self.y_idx] = self.df[self.y_idx] - y_offset



    def _init_lane_color_map(self, cmap = cm.tab10):
        '''初始化车道颜色映射。
        如不需要对车道指定颜色, 可关闭。
        '''
        unique_lanes = self.df[self.lane_idx].unique()
        self.lane_color_map = {lane: cmap(i) for i, lane in enumerate(unique_lanes)}
