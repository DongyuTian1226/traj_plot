import os
from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from research_plt import ResearchPlt


class TrajOnImagePlotter(ResearchPlt):
    '''轨迹点在图像上绘制, 继承自ResearchPlt类

    在一张图像上, 将轨迹点绘制成散点图或线条图。可用于论文图片展示
    '''
    def __init__(
            self,
            data_path: str,
            img_path: str,
            time_idx: Union[int, str],
            car_idx: Union[int, str],
            x_idx: Union[int, str],
            y_idx: Union[int, str],
            save_dir: str = None,
            max_time: int = 1e10,
            ids: Union[int, str, list] = None,
            **kwargs,
            ):
        '''
        input
        -----
        data_path: str, 轨迹文件路径
        img_path: str, 图像路径
        time_idx: Union[int, str], 时间所在列的索引或列名
        car_idx: Union[int, str], ID所在列的索引或列名
        x_idx: Union[int, str], 横坐标所在列的索引或列名
        y_idx: Union[int, str], 纵坐标所在列的索引或列名
        **kwargs: ResearchPlt的初始化参数, 参见ResearchPlt
        '''
        super().__init__(**kwargs)
        self.path = data_path
        self.img_path = img_path
        self.save_dir = save_dir
        self.df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_excel(data_path)
        self.time_idx = time_idx if isinstance(time_idx, str) else self.df.columns[time_idx]
        self.car_idx = car_idx if isinstance(car_idx, str) else self.df.columns[car_idx]
        self.x_idx = x_idx if isinstance(x_idx, str) else self.df.columns[x_idx]
        self.y_idx = y_idx if isinstance(y_idx, str) else self.df.columns[y_idx]
        # 数据预处理
        self.df = self.df[self.df[self.time_idx] <= max_time]
        if ids is not None:
            self.df = self.df[self.df[self.car_idx].isin(ids)]
        self.df = self.df.sort_values(by=[self.time_idx, self.car_idx], axis=0, ascending=[True, True])
        self.df = self.df.reset_index(drop=True)

    def run(
            self,
            x_offset: int = 0, y_offset: int = 0,
            mask_color: np.ndarray = None, mask_alpha: float = 0.5,
            if_line: bool = False, line_width: float = 2, line_style: str = '-', line_alpha: float = 0.5,
            if_scatter: bool = False, marker: str = 'o', markersize: float = 3, marker_alpha: float = 0.8,
            line_color: str = 'red', scatter_color: str = 'red',
            ):
        '''function run
        以散点/线段形式绘制轨迹点在图像上, 可以设置散点/线段颜色, 大小和透明度, 蒙版颜色和透明度

        input
        -----
        x_offset, y_offset: int, 图像左上角坐标偏移量, 默认为0
        if_line: bool, 是否绘制线条, 默认False
        line_width, line_style, line_alpha: float, str, 线条宽度, 线条样式, 线条透明度
        if_scatter: bool, 是否绘制散点, 默认False
        marker, markersize, marker_alpha: str, float, float, 散点样式, 散点大小, 散点透明度
        line_color, scatter_color: str, 线条颜色, 散点颜色
        mask_color: np.ndarray, 蒙版颜色,格式为(R, G, B, A), 不设置则不显示蒙版
        mask_alpha: float, 蒙版透明度
        '''
        # prepare
        if not if_line and not if_scatter:
            raise ValueError('if_line and if_scatter cannot both be False. 要不你画啥呢！')
        self.df[self.x_idx] = self.df[self.x_idx] + x_offset
        self.df[self.y_idx] = self.df[self.y_idx] + y_offset
        # 读取图像
        img = Image.open(self.img_path)
        height, width = img.size
        plt.figure()
        plt.imshow(img)
        # 绘制蒙版
        if mask_color:
            _ = plt.imshow(np.ones((width, height, 4)) * mask_color, alpha=mask_alpha)
        # 绘制散点图
        if if_scatter:
            plt.scatter(self.df[self.x_idx], self.df[self.y_idx],
                        c=scatter_color, s=markersize, alpha=marker_alpha, marker=marker)
        # 绘制线条图
        if if_line:
            # 遍历每个ID，按时间顺序连接点
            for _, car_traj in self.df.groupby(self.car_idx):
                plt.plot(car_traj[self.x_idx], car_traj[self.y_idx],
                        c=line_color, linewidth=line_width, linestyle=line_style, alpha=line_alpha)
        # 显示结果
        plt.axis('off')  # 关闭坐标轴
        # 根据绘制类型确定保存文件名
        if self.save_dir is not None:
            save_path = os.path.join(self.save_dir + os.path.basename(self.path))
        else:
            save_path = self.path.split('.')[0]
            save_path = save_path + '_scatter' if if_scatter else save_path
            save_path = save_path + '_line' if if_line else save_path
            save_path = save_path + '.png'
        plt.savefig(save_path)
        plt.close()
        # end: restore data
        self.df[self.x_idx] = self.df[self.x_idx] - x_offset
        self.df[self.y_idx] = self.df[self.y_idx] - y_offset


def main_hksts_debris_paper():
    '''抛洒物画图, HKSTS论文'''
    x_idx = 3
    y_idx = 4
    time_idx = 0
    car_idx = 2
    scatter_color = 'yellow'
    markersize = 0.2
    marker_alpha = 0.3
    line_color = (139/255, 69/255, 19/255, 1)       # white
    line_width = 1
    line_alpha = 0

    path_list = [
        r'D:\东南大学\科研\基于轨迹的抛洒物检测\codes\yolo-detect\chair1\chair - Trim1.csv',
    ]
    img_path_list = [
        r'F:\debris-images\chair - Trim1\frame_0.jpg',
    ]
    # 画图
    num = len(path_list)
    for i in range(num):
        path = path_list[i]
        img_path = img_path_list[i]
        toip = TrajOnImagePlotter(path, img_path, x_idx, y_idx, time_idx, car_idx)
        toip.plot_scatter(scatter_color, markersize, marker_alpha)
        toip.plot_line(line_color, line_width, line_alpha)

if __name__ == '__main__':
    main_hksts_debris_paper()
