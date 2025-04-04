import matplotlib.pyplot as plt


class ResearchPlt:
    '''
    更符合科研作图的plt画图, 使用方式：
    1. 继承该类(推荐)
    2. 在对应代码文件开头处, 导入并运行该类, 实现全局配置
    '''
    def __init__(
            self,
            backend='agg',
            figsize=(12, 8),
            dpi=300,
            bbox_inches='tight',
            font='Times New Roman',
            font_size=16,
            legend_framealpha=0.25,
            legend_loc='upper right',
            legend_handletextpad=0,
            ):
        '''设置plt画图的全局参数'''
        # 非交互式模式, 绘图速度更快
        plt.switch_backend(backend)
        # 画布
        plt.rcParams['figure.figsize'] = figsize        # 画布大小
        plt.rcParams['figure.dpi'] = dpi                # 画布分辨率
        plt.rcParams['savefig.bbox'] = bbox_inches      # 保存图像时, 去掉多余空白
        # 字体
        plt.rcParams['font.sans-serif'] = [font]        # 字体
        plt.rcParams['font.size'] = font_size           # 字体大小
        plt.rcParams['axes.unicode_minus'] = False      # 解决负号'-'显示为方块的问题
        # legend
        plt.rcParams['legend.framealpha'] = legend_framealpha           # legend透明
        plt.rcParams['legend.loc'] = legend_loc                         # legend位置
        plt.rcParams['legend.handletextpad'] = legend_handletextpad     # legend图例文字间距

    def show_legend_sorted(self, title: str):
        '''
        处理legend的显示顺序, 使其按照顺序排列
        input
        -----
        title: str, legend标题
        '''
        handles, labels = plt.gca().get_legend_handles_labels()
        sorted_labels = sorted(labels)
        sorted_handles = [handles[labels.index(label)] for label in sorted_labels]
        _ = plt.legend(sorted_handles, sorted_labels, title=title)

    def xy_limit_with_gap(
            self,
            x_min: float = None, x_max: float = None, x_gap: float = None,
            y_min: float = None, y_max: float = None, y_gap: float = None,
            ):
        '''
        设置x轴和y轴的范围, 以及前后给画图留出一定间隔。最终图像呈现的范围为
        [x_min - x_gap, x_max + x_gap]和[y_min - y_gap, y_max + y_gap]

        input
        -----
        x_min, x_max: float, x轴范围
        x_gap: float, x轴范围前后间隔
        y_min, y_max: float, y轴范围
        y_gap: float, y轴范围前后间隔
        '''
        if x_min and x_max:
            x_gap = x_gap or 0
            plt.xlim(x_min - x_gap, x_max + x_gap)
        if y_min and y_max:
            y_gap = y_gap or 0
            plt.ylim(y_min - y_gap, y_max + y_gap)

    def specified_grid(
            self,
            x_grid: list = None, x_grid_color: str = 'grey',
            x_grid_style: str = '--', x_grid_width: float = 0.5,
            y_grid: list = None, y_grid_color: str = 'black',
            y_grid_style: str = '-', y_grid_width: float = 0.5,
            ):
        '''
        设置指定的网格线, 颜色, 样式, 宽度
        input
        -----
        x_grid: list, 竖线网格线的x位置
        y_grid: list, 横线网格线的y位置
        x_grid_color, y_grid_color: str, 网格线颜色
        x_grid_style, y_grid_style: str, 网格线样式
        x_grid_width, y_grid_width: float, 网格线宽度
        '''
        for vertical_line in x_grid or []:
            plt.axvline(x=vertical_line, color=x_grid_color,
                        linestyle=x_grid_style, linewidth=x_grid_width)
        for horizontal_line in y_grid or []:
            plt.axhline(y=horizontal_line, color=y_grid_color,
                        linestyle=y_grid_style, linewidth=y_grid_width)


class ExamplePloter(ResearchPlt):
    '''示例, 基于ResearchPlt类进行继承'''
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path
