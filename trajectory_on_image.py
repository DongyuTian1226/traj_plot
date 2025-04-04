from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from research_plt import ResearchPlt


class TrajOnImagePlotter(ResearchPlt):
    '''轨迹点在图像上绘制, 继承自ResearchPlt类

    在一张图像上, 将轨迹点绘制成散点图或线条图
    '''
    def __init__(
            self,
            path: str,
            img_path: str,
            x_idx: Union[int, str],
            y_idx: Union[int, str],
            time_idx: Union[int, str],
            car_idx: Union[int, str],
            **kwargs,
            ):
        '''
        input
        -----
        path: str, 轨迹文件路径
        img_path: str, 图像路径
        x_idx: Union[int, str], 横坐标所在列的索引或列名
        y_idx: Union[int, str], 纵坐标所在列的索引或列名
        time_idx: Union[int, str], 时间所在列的索引或列名
        car_idx: Union[int, str], ID所在列的索引或列名
        **kwargs: ResearchPlt的初始化参数, 参见ResearchPlt
        '''
        super().__init__(**kwargs)
        self.path = path
        self.img_path = img_path
        df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
        self.x_idx = x_idx if isinstance(x_idx, str) else df.columns[x_idx]
        self.y_idx = y_idx if isinstance(y_idx, str) else df.columns[y_idx]
        self.time_idx = time_idx if isinstance(time_idx, str) else df.columns[time_idx]
        self.car_idx = car_idx if isinstance(car_idx, str) else df.columns[car_idx]
        df.sort_values(by=[self.car_idx, self.time_idx])
        self.df = df

    def plot_scatter(
            self,
            scatter_color: str = 'red',
            scatter_size: float = 3,
            scatter_alpha: float = 0.8,
            mask_color: np.ndarray = None,
            mask_alpha: float = 0.5,
            ):
        '''function plot_scatter_on_image
        以散点形式绘制轨迹点在图像上, 可以设置散点颜色, 大小和透明度, 蒙版颜色和透明度
        input
        -----
        scatter_color: str, 散点颜色
        scatter_size: float, 散点大小
        scatter_alpha: float, 散点透明度
        mask_color: np.ndarray, 蒙版颜色,格式为(R, G, B, A), 不设置则不显示蒙版
        mask_alpha: float, 蒙版透明度
        '''
        # 读取图像
        img = Image.open(self.img_path)
        height, width = img.size
        plt.figure()
        plt.imshow(img)
        if mask_color:
            _ = plt.imshow(np.ones((width, height, 4)) * mask_color, alpha=mask_alpha)

        # 在图像上绘制散点图
        plt.scatter(self.df[self.x_idx], self.df[self.y_idx],
                    c=scatter_color, s=scatter_size, alpha=scatter_alpha)

        # 显示结果
        plt.axis('off')  # 关闭坐标轴
        save_path = self.path.split('.')[0] + '_scatter.png'
        plt.savefig(save_path)
        plt.close()

    def plot_line(
            self,
            line_color: str = 'red',
            line_width: float = 2,
            line_alpha: float = 0.5,
            mask_color: np.ndarray = None,
            mask_alpha: float = 0.5,
            ):
        '''function plot_lines_on_image
        以线条形式绘制轨迹点在图像上, 可以设置线条颜色, 宽度和透明度, 蒙版颜色和透明度
        input
        ------
        line_color: str, 线条颜色
        line_width: float, 线条宽度
        line_alpha: float, 线条透明度
        mask_color: np.ndarray, 蒙版颜色,格式为(R, G, B, A), 不设置则不显示蒙版
        mask_alpha: float, 蒙版透明度
        '''
        # 读取图像
        img = Image.open(self.img_path)
        height, width = img.size
        plt.figure()
        plt.imshow(img)
        if mask_color:
            _ = plt.imshow(np.ones((width, height, 4)) * mask_color, alpha=mask_alpha)

        # 遍历每个ID，按时间顺序连接点
        for _, car_traj in self.df.groupby(self.car_idx):
            plt.plot(car_traj[self.x_idx], car_traj[self.y_idx],
                     c=line_color, linewidth=line_width, alpha=line_alpha)

        # 显示结果
        plt.axis('off')  # 关闭坐标轴
        save_path = self.path.split('.')[0] + '_line.png'
        plt.savefig(save_path)
        plt.close()


def main_hksts_debris_paper():
    '''抛洒物画图, HKSTS论文'''
    x_idx = 3
    y_idx = 4
    time_idx = 0
    car_idx = 2
    scatter_color = 'yellow'
    scatter_size = 0.2
    scatter_alpha = 0.3
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
        toip.plot_scatter(scatter_color, scatter_size, scatter_alpha)
        toip.plot_line(line_color, line_width, line_alpha)

if __name__ == '__main__':
    main_hksts_debris_paper()
