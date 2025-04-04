import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_scatter_on_image(csv_path: str, x_col_idx: int, y_col_idx: int,
                          img_path: str, scatter_color: str = 'red',
                          scatter_size: float = 3,
                          scatter_alpha: float = 0.8,
                          mask_color: np.ndarray = None, mask_alpha: float = 0.5):
    '''function plot_scatter_on_image

    input
    ------
    csv_path: str, 轨迹点的CSV文件路径
    x_col_idx: int, 横坐标所在列的索引
    y_col_idx: int, 纵坐标所在列的索引
    img_path: str, 图像路径
    scatter_color: str, 散点颜色
    scatter_size: float, 散点大小
    scatter_alpha: float, 散点透明度
    mask_color: np.ndarray, 蒙版颜色,格式为(R, G, B, A)
    mask_alpha: float, 蒙版透明度

    return
    ------
    None

    在一张图像上, 将csv文件中的轨迹点绘制成散点图
    '''
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 提取横纵坐标
    x = df.iloc[:, x_col_idx]
    y = df.iloc[:, y_col_idx]

    # 读取图像
    img = Image.open(img_path)
    height, width = img.size
    plt.figure()
    plt.tight_layout()
    plt.imshow(img)
    if mask_color:
        mask = plt.imshow(np.ones((width, height, 4)) * mask_color, alpha=mask_alpha)

    # 在图像上绘制散点图
    plt.scatter(x, y, c=scatter_color, s=scatter_size, alpha=scatter_alpha)

    # 显示结果
    plt.axis('off')  # 关闭坐标轴
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_path = csv_path.replace('.csv', '.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    _drop_image_boundary(save_path)


def plot_lines_on_image(csv_path: str, x_col_idx: int, y_col_idx: int,
                        time_col_idx: int, id_col_idx: int, img_path: str,
                        line_color: str = 'red', line_width: float = 2,
                        line_alpha: float = 0.5,
                        mask_color: np.ndarray = None, mask_alpha: float = 0.5):
    '''function plot_lines_on_image

    input
    ------
    csv_path: str, 轨迹点的CSV文件路径
    x_col_idx: int, 横坐标所在列的索引
    y_col_idx: int, 纵坐标所在列的索引
    time_col_idx: int, 时间所在列的索引
    id_col_idx: int, ID所在列的索引
    img_path: str, 图像路径
    line_color: str, 线条颜色
    line_width: float, 线条宽度
    line_alpha: float, 线条透明度
    mask_color: np.ndarray, 蒙版颜色, 格式为(R, G, B, A)
    mask_alpha: float, 蒙版透明度

    return
    ------
    None

    在一张图像上, 将csv文件中的轨迹点按时间顺序连接成线
    '''
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 根据ID和时间排序
    df = df.sort_values(by=[df.columns[id_col_idx], df.columns[time_col_idx]])

    # 读取图像
    img = Image.open(img_path)
    height, width = img.size
    plt.figure()
    plt.imshow(img)
    if mask_color:
        mask = plt.imshow(np.ones((width, height, 4)) * mask_color, alpha=mask_alpha)

    # 获取唯一的ID列表
    unique_ids = df.iloc[:, id_col_idx].unique()

    # 遍历每个ID，按时间顺序连接点
    for uid in unique_ids:
        target_points = df[df.iloc[:, id_col_idx] == uid]
        x = target_points.iloc[:, x_col_idx]
        y = target_points.iloc[:, y_col_idx]
        plt.plot(x, y, c=line_color, linewidth=line_width, alpha=line_alpha)
    
    # 显示结果
    plt.axis('off')  # 关闭坐标轴
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_path = csv_path.replace('.csv', '_line.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    _drop_image_boundary(save_path)


def _drop_image_boundary(img_path: str, w_bound: int = 30, h_bound: int = 30):
    '''function _drop_image_boundary

    input
    ------
    img_path: str, 图像路径
    w_bound: int, 宽度边界, 单位像素
    h_bound: int, 高度边界, 单位像素

    return
    ------
    None

    将图像的边界裁剪掉
    '''
    img = Image.open(img_path)
    w, h = img.size
    img = img.crop((w_bound, h_bound, w-w_bound, h-h_bound))
    # 保存
    img.save(img_path)


if __name__ == '__main__':
    # 调用函数示例
    # csv_path = r'D:\东南大学\科研\本科阶段2019-2023.6\毕设202306\f 项目代码\yolov5-6.2 - mask - track\runs\detect\exp5\DDK65+966-down-CAM24_dvr_C1R27P472_15-51-20_15-55-59_0 - Trim1直接主函数出_filtered_smoothed_kalman.csv'
    x_col_idx = 3
    y_col_idx = 4
    time_col_idx = 0
    id_col_idx = 2
    # img_path = r'F:\debris-images\K65+966-down-CAM24_dvr_C1R27P472_15-51-20_15-55-59_0\frame_0.jpg'
    scatter_color = 'yellow'
    scatter_size = 0.2
    scatter_alpha = 0.3
    # plot_scatter_on_image(csv_path, x_col_idx, y_col_idx, img_path, scatter_color, scatter_size, scatter_alpha)
    line_color = (139/255, 69/255, 19/255, 1)       # white
    line_width = 1
    line_alpha = 0
    # plot_lines_on_image(csv_path, x_col_idx, y_col_idx, time_col_idx, id_col_idx, img_path, line_color, line_width, line_alpha)

    # 画多个图像
    csv_path_list = [
        r'D:\东南大学\科研\基于轨迹的抛洒物检测\codes\yolo-detect\53+665\K53+665-downCAM67_dvr_C1R64P434_14-26-00_14-31-00_0直接主函数出_filtered_smoothed_kalman.csv',
        r'D:\东南大学\科研\基于轨迹的抛洒物检测\codes\yolo-detect\65+966\K65+966-down-CAM24_dvr_C1R27P472_15-51-20_15-55-59_0 - Trim1直接主函数出_filtered_smoothed_kalman.csv',
        r'D:\东南大学\科研\基于轨迹的抛洒物检测\codes\yolo-detect\71+616\K71+616-up-CAM20_dvr_C1R27P478_11-30-30_11-39-59_0直接主函数出_filtered_smoothed_kalman.csv',
        r'D:\东南大学\科研\基于轨迹的抛洒物检测\codes\yolo-detect\75+001\K75+001-down-CAM18_dvr_C1R27P484_15-33-05_15-43-59_0直接主函数出_filtered_smoothed_kalman.csv',
        r'D:\东南大学\科研\基于轨迹的抛洒物检测\codes\yolo-detect\chair1\chair - Trim1直接主函数出_filtered_smoothed_kalman.csv',
        r'D:\东南大学\科研\基于轨迹的抛洒物检测\codes\yolo-detect\chair2\chair - Trim2直接主函数出_filtered_smoothed_kalman.csv',
        r'D:\东南大学\科研\基于轨迹的抛洒物检测\codes\yolo-detect\shipin5+700\shipin5+700直接主函数出_filtered_smoothed_kalman.csv',
        r'D:\东南大学\科研\基于轨迹的抛洒物检测\codes\yolo-detect\shipin32+100\shipinK32+100直接主函数出_filtered_smoothed_kalman.csv',
        r'D:\东南大学\科研\基于轨迹的抛洒物检测\codes\yolo-detect\shipin31100\shipin31100直接主函数出_filtered_smoothed_kalman.csv',
        r'D:\东南大学\科研\基于轨迹的抛洒物检测\codes\yolo-detect\shipin32100\shipinK32100直接主函数出_filtered_smoothed_kalman.csv',
        r'D:\东南大学\科研\基于轨迹的抛洒物检测\codes\yolo-detect\shipin536654\shipink536654直接主函数出_filtered_smoothed_kalman.csv',
        r'D:\东南大学\科研\基于轨迹的抛洒物检测\codes\yolo-detect\spill debris\spilled debris - Trim1直接主函数出_filtered_smoothed_kalman.csv',
    ]
    img_path_list = [
        r'F:\debris-images\K53+665-downCAM67_dvr_C1R64P434_14-26-00_14-31-00_0\frame_0.jpg',
        r'F:\debris-images\K65+966-down-CAM24_dvr_C1R27P472_15-51-20_15-55-59_0\frame_0.jpg',
        r'F:\debris-images\K71+616-up-CAM20_dvr_C1R27P478_11-30-30_11-39-59_0\frame_0.jpg',
        r'F:\debris-images\K75+001-down-CAM18_dvr_C1R27P484_15-33-05_15-43-59_0\frame_0.jpg',
        r'F:\debris-images\chair - Trim1\frame_0.jpg',
        r'F:\debris-images\chair - Trim2\frame_0.jpg',
        r'F:\debris-images\shipin5+700\frame_0.jpg',
        r'F:\debris-images\shipinK32+100\frame_0.jpg',
        r'F:\debris-images\shipin31100\frame_0.jpg',
        r'F:\debris-images\shipinK32100\frame_0.jpg',
        r'F:\debris-images\shipink536654\frame_0.jpg',
        r'F:\debris-images\spilled debris - Trim1\frame_0.jpg',
    ]
    for i in range(len(csv_path_list)):
        csv_path = csv_path_list[i]
        img_path = img_path_list[i]
        # plot_scatter_on_image(csv_path, x_col_idx, y_col_idx, img_path, scatter_color, scatter_size, scatter_alpha)
        plot_lines_on_image(csv_path,
                            x_col_idx, y_col_idx, time_col_idx, id_col_idx,
                            img_path,
                            line_color, line_width, line_alpha,
                            mask_color=(1, 1, 1, 1), mask_alpha=0.5)
