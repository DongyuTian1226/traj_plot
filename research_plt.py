import matplotlib.pyplot as plt
import numpy as np

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
            font=['Times New Roman', 'SimSun'],
            font_size=16,
            legend_framealpha=0.25,
            legend_loc='upper right',
            legend_handletextpad=0,
            ):
        '''设置plt画图的全局参数, 包括绘图后端, 画布参数, 字体参数, legend参数等

        input
        -----
        backend: str, 绘图后端, 默认为'agg', 即非交互式模式, 绘图速度更快
        figsize: tuple, 画布大小, 默认为(12, 8)
        dpi: int, 画布分辨率, 默认为300
        bbox_inches: str, 保存图像时, 去掉多余空白, 默认为'tight'
        font: str | list, 字体, 默认为'Times New Roman', 'SimSun'
        font_size: int, 字体大小, 默认为16
        legend_framealpha: float, legend透明, 默认为0.25
        legend_loc: str, legend位置, 默认为'upper right'
        legend_handletextpad: float, legend图例文字间距, 默认为0
        '''
        # 非交互式模式, 绘图速度更快
        # plt.switch_backend(backend)
        # 画布
        plt.rcParams['figure.figsize'] = figsize        # 画布大小
        plt.rcParams['figure.dpi'] = dpi                # 画布分辨率
        plt.rcParams['savefig.bbox'] = bbox_inches      # 保存图像时, 去掉多余空白
        # 字体
        plt.rcParams['font.family'] = font        # 字体
        plt.rcParams['font.size'] = font_size           # 字体大小
        plt.rcParams['axes.unicode_minus'] = False      # 解决负号'-'显示为方块的问题
        # legend
        plt.rcParams['legend.framealpha'] = legend_framealpha           # legend透明
        plt.rcParams['legend.loc'] = legend_loc                         # legend位置
        plt.rcParams['legend.handletextpad'] = legend_handletextpad     # legend图例文字间距

    def show_legend_sorted(self, title: str = None):
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
        x_min: float = None, x_max: float = None, x_gap: float = 0,
        y_min: float = None, y_max: float = None, y_gap: float = 0,
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
        x_min = x_min - x_gap if x_min is not None else None
        x_max = x_max + x_gap if x_max is not None else None
        y_min = y_min - y_gap if y_min is not None else None
        y_max = y_max + y_gap if y_max is not None else None
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    def xy_grid(
        self,
        x_grid: list = None, x_grid_color: str = 'grey', x_grid_style: str = '--', x_grid_width: float = 0.5,
        y_grid: list = None, y_grid_color: str = 'black', y_grid_style: str = '-', y_grid_width: float = 0.5,
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

    def show_colorbar_speed(
        self, v_min: int = 0, v_max: int = 120, v_step: int = 20,
        cmap: str = 'jet_r', label: str = None):
        '''
        绘制车辆速度的colorbar
        '''
        plt.clim(v_min, v_max)
        cbar = plt.colorbar(cmap=cmap, shrink=1, aspect=30)
        cbar.set_ticks(range(v_min, v_max + v_step, v_step))
        if label is not None:
            cbar.set_label(label)

    def one_call_xy_settings(
        self,
        x_min: float = None, x_max: float = None, x_gap: float = 0,
        y_min: float = None, y_max: float = None, y_gap: float = 0,
        x_grid: list = None, x_grid_color: str = 'grey', x_grid_style: str = '--', x_grid_width: float = 0.5,
        y_grid: list = None, y_grid_color: str = 'black', y_grid_style: str = '-', y_grid_width: float = 0.5,
        ):
        '''
        集成self.xy_limit_with_gap和self.xy_grid函数。
        '''
        self.xy_limit_with_gap(x_min, x_max, x_gap, y_min, y_max, y_gap)
        self.xy_grid(x_grid, x_grid_color, x_grid_style, x_grid_width,
                     y_grid, y_grid_color, y_grid_style, y_grid_width)

    @staticmethod
    def setup_click_handler(fig, ax, scatter_objects, ids, dist_thres: float = 5):
        """
        TODO 一些场景可能会存在偏移显示，但不多，调小dist_thres可规避这个问题
        为散点图设置点击事件处理器（每组散点拥有相同ID）
        
        参数:
            fig: matplotlib的figure对象
            ax: 绘图的axes对象
            scatter_objects: 散点对象列表，每个对象对应一组轨迹点
            ids: ID列表，与散点对象列表一一对应，每个元素是该组散点的ID
            dist_thres: float, 点击距离阈值
        """
        # 用于存储当前显示的标注
        current_annotation = None
        
        def on_click(event):
            nonlocal current_annotation
            
            # 如果点击不在坐标轴区域内，则忽略
            if event.inaxes != ax:
                return
            
            # 获取点击位置坐标
            click_x, click_y = event.xdata, event.ydata
            print(f"点击位置为({click_x}s, {click_y}m)")
            # 遍历所有散点对象，查找被点击的点
            for scatter, vehicle_id in zip(scatter_objects, ids):
                # 获取该散点对象的所有点坐标
                points = scatter.get_offsets()
                if len(points) == 0:
                    continue
                    
                # 计算点击位置与每个点的距离
                distances = np.sqrt((points[:, 0] - click_x)**2 + (points[:, 1] - click_y)** 2)
                
                # 找到最近的点
                min_dist = np.min(distances)
                
                # 设置距离阈值，可根据你的数据尺度调整
                if min_dist < dist_thres:  # 阈值越小，需要点击越精确
                    # 在控制台打印ID信息
                    print(f"点击了车辆 ID: {vehicle_id}")
                    
                    # 移除之前的标注
                    if current_annotation:
                        current_annotation.remove()
                    
                    # 找到最近点的坐标用于显示标注
                    closest_point_idx = np.argmin(distances)
                    closest_x, closest_y = points[closest_point_idx]
                    
                    # 创建新标注
                    current_annotation = ax.annotate(
                        f"ID: {vehicle_id}",
                        (closest_x, closest_y),
                        xytext=(5, 5),  # 文本相对点的偏移量
                        textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1")
                    )
                    
                    # 更新图形
                    fig.canvas.draw_idle()
                    break
        
        # 连接事件处理函数
        fig.canvas.mpl_connect('button_press_event', on_click)
        print("已启用点击交互功能，点击散点可查看车辆ID")
