import numpy as np
import matplotlib.pyplot as plt

def setup_click_handler(fig, ax, scatter_objects, ids):
    """
    为散点图设置点击事件处理器（每组散点拥有相同ID）
    
    参数:
        fig: matplotlib的figure对象
        ax: 绘图的axes对象
        scatter_objects: 散点对象列表，每个对象对应一组轨迹点
        ids: ID列表，与散点对象列表一一对应，每个元素是该组散点的ID
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
            if min_dist < 10:  # 阈值越小，需要点击越精确
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

# ----------------------
# 使用示例（集成到你的代码中）
# ----------------------
if __name__ == "__main__":
    # 这里仅作为示例，替换为你的实际数据和绘图代码
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 示例：每组散点拥有相同ID
    scatter_objs = []
    vehicle_ids = []  # 每个元素对应一组散点的ID
    
    # 第一组数据 - ID: V001
    x1 = np.random.rand(15) * 100
    y1 = np.random.rand(15) * 100 + 50  # 加点偏移使其分开
    scatter1 = ax.scatter(x1, y1, c='blue', label='V001', alpha=0.7)
    scatter_objs.append(scatter1)
    vehicle_ids.append("V001")  # 整组共用一个ID
    
    # 第二组数据 - ID: V002
    x2 = np.random.rand(12) * 100
    y2 = np.random.rand(12) * 100
    scatter2 = ax.scatter(x2, y2, c='red', label='V002', alpha=0.7)
    scatter_objs.append(scatter2)
    vehicle_ids.append("V002")  # 整组共用一个ID
    
    # 第三组数据 - ID: V003
    x3 = np.random.rand(10) * 100 + 20  # 加点偏移使其分开
    y3 = np.random.rand(10) * 100 + 30
    scatter3 = ax.scatter(x3, y3, c='green', label='V003', alpha=0.7)
    scatter_objs.append(scatter3)
    vehicle_ids.append("V003")  # 整组共用一个ID
    
    # 设置图例和标题
    ax.legend(title="车辆ID")
    ax.set_title('车辆轨迹散点图 - 点击点查看ID')
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 关键步骤：设置点击事件处理器
    setup_click_handler(fig, ax, scatter_objs, vehicle_ids)
    
    plt.show()
