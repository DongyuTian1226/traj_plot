from typing import Union
import pandas as pd
import matplotlib.pyplot as plt

from research_plt import ResearchPlt


class TrajXyPlot(ResearchPlt):
    '''轨迹点在xy平面上绘制, 继承自ResearchPlt类'''
    def __init__(
            self,
            path: str,
            x_idx: Union[int, str],
            y_idx: Union[int, str],
            lane_idx: Union[int, str],
            save_path: str = None,
            **kwagrgs):
        '''继承自ResearchPlt类, 画轨迹点在xy平面上的散点图
        
        input

        '''
        super().__init__(**kwagrgs)
        self.path = path
        self.save_path = save_path
        self.df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
        self.x_idx = x_idx if isinstance(x_idx, str) else self.df.columns[x_idx]
        self.y_idx = y_idx if isinstance(y_idx, str) else self.df.columns[y_idx]
        self.lane_idx = lane_idx if isinstance(lane_idx, str) else self.df.columns[lane_idx]

    def run(
            self,
            scatter_size: float = 1,
            scatter_alpha: float = 0.5,
            origin: bool = True,
            origin_size: float = 100,
            origin_color: str = 'red',
            origin_marker: str = 'o',
            ):
        '''运行画图'''
        plt.figure()
        if origin:
            plt.scatter([0], [0], s=origin_size, c=origin_color, marker=origin_marker)
        for lane, lane_data in self.df.groupby(self.lane_idx):
            # 加alpha会变糊
            plt.scatter(lane_data[self.x_idx], lane_data[self.y_idx],
                        label='lane'+str(lane), s=scatter_size, alpha=scatter_alpha)

        # 添加元素
        plt.xlabel(self.x_idx)
        plt.ylabel(self.y_idx)
        self.show_legend_sorted(title=self.lane_idx)
        save_path = self.save_path or self.path.split('.')[0] + '.png'
        plt.savefig(save_path)
        plt.close()


def main_raoyue():
    '''绕越高速雷达数据平面图'''
    # field data
    x_idx = 3
    y_idx = 4
    lane_idx = 2

    # # 画单个文件
    path = r"D:\myscripts\spill-detection\data\extractedData\2024-3-27-17_byDevice\K78+760_1.csv"
    txyp = TrajXyPlot(path=path, x_idx=x_idx, y_idx=y_idx, lane_idx=lane_idx)
    txyp.run()


if __name__ == '__main__':
    main_raoyue()
