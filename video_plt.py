import os
from tqdm import tqdm
from typing import Union
import pandas as pd
import numpy as np
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt

from research_plt import ResearchPlt


class TrajVideoPlot(ResearchPlt):
    '''轨迹点在xy平面上绘制并生成视频, 继承自ResearchPlt类
    '''
    def __init__(
            self,
            path: str,
            time_idx: Union[int, str],
            car_idx: Union[int, str],
            x_idx: Union[int, str],
            y_idx: Union[int, str],
            vx_idx: Union[int, str],
            vy_idx: Union[int, str],
            save_dir: str = None,
            fps: int = 30,
            video_width: int = 1920,
            video_height: int = 1080,
            max_frames: int = None,
            **kwargs):
        '''继承自ResearchPlt类, 生成轨迹点的上帝视角监控视频

        input
        -----
        path: str, 轨迹文件路径
        time_idx: Union[int, str], 时间/帧所在列的索引或列名
        car_idx: Union[int, str], 车辆id所在列的索引或列名
        x_idx: Union[int, str], 横坐标所在列的索引或列名
        y_idx: Union[int, str], 纵坐标所在列的索引或列名
        vx_idx: Union[int, str], x方向速度所在列的索引或列名
        vy_idx: Union[int, str], y方向速度所在列的索引或列名
        save_dir: str, 保存视频的文件夹, 如果为None则保存到path所在文件夹
        fps: int, 视频帧率, 默认为30
        video_width: int, 视频宽度, 默认为1920
        video_height: int, 视频高度, 默认为1080
        max_frames: int, 最大处理帧数, 默认为None（不设上限）
        '''
        super().__init__(**kwargs)
        self.path = path
        self.fps = fps
        self.video_width = video_width
        self.video_height = video_height
        self.max_frames = max_frames
        
        # 创建输出文件夹（直接保存在与csv文件同级的文件夹）
        self.save_dir = save_dir or os.path.dirname(path)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存路径信息
        self.path = path
        
        # 初始化索引参数
        self.time_idx = time_idx
        self.car_idx = car_idx
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.vx_idx = vx_idx
        self.vy_idx = vy_idx
        
        # 读取数据或使用传入的df
        if path is not None:
            self.df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
            # 如果传入的是索引，转换为列名
            if not isinstance(self.time_idx, str):
                self.time_idx = self.df.columns[self.time_idx]
            if not isinstance(self.car_idx, str):
                self.car_idx = self.df.columns[self.car_idx]
            if not isinstance(self.x_idx, str):
                self.x_idx = self.df.columns[self.x_idx]
            if not isinstance(self.y_idx, str):
                self.y_idx = self.df.columns[self.y_idx]
            if not isinstance(self.vx_idx, str):
                self.vx_idx = self.df.columns[self.vx_idx]
            if not isinstance(self.vy_idx, str):
                self.vy_idx = self.df.columns[self.vy_idx]
            
            # 数据预处理
            self.df['speed'] = np.sqrt(self.df[self.vx_idx]**2 + self.df[self.vy_idx]**2)
            self.df = self.df.sort_values(by=[self.time_idx, self.car_idx], axis=0, ascending=[True, True])
            self.df = self.df.reset_index(drop=True)
            
            # 计算坐标范围，用于缩放
            self.x_min = self.df[self.x_idx].min()
            self.x_max = self.df[self.x_idx].max()
            self.y_min = self.df[self.y_idx].min()
            self.y_max = self.df[self.y_idx].max()
            self.x_range = self.x_max - self.x_min
            self.y_range = self.y_max - self.y_min
            
            # 计算速度范围，用于颜色映射
            self.speed_min = self.df['speed'].min()
            self.speed_max = self.df['speed'].max()
            
            # 颜色映射
            self.cmap = cm.turbo
            self.norm = plt.Normalize(self.speed_min, self.speed_max)
            
            # 获取所有帧
            self.frames = sorted(self.df[self.time_idx].unique())
            # 应用最大帧数限制
            if self.max_frames is not None:
                self.frames = self.frames[:self.max_frames]
        else:
            # 当path为None时，延迟初始化，等待后续设置df
            pass
    
    def _recalculate_attributes(self):
        '''重新计算必要的属性，当直接设置df时使用'''  
        # 数据预处理
        if 'speed' not in self.df.columns:
            self.df['speed'] = np.sqrt(self.df[self.vx_idx]**2 + self.df[self.vy_idx]**2)
        self.df = self.df.sort_values(by=[self.time_idx, self.car_idx], axis=0, ascending=[True, True])
        self.df = self.df.reset_index(drop=True)
        
        # 计算坐标范围，用于缩放
        self.x_min = self.df[self.x_idx].min()
        self.x_max = self.df[self.x_idx].max()
        self.y_min = self.df[self.y_idx].min()
        self.y_max = self.df[self.y_idx].max()
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        
        # 计算速度范围，用于颜色映射
        self.speed_min = self.df['speed'].min()
        self.speed_max = self.df['speed'].max()
        
        # 颜色映射
        self.cmap = cm.turbo
        self.norm = plt.Normalize(self.speed_min, self.speed_max)
        
        # 获取所有帧
        self.frames = sorted(self.df[self.time_idx].unique())
        # 应用最大帧数限制
        if self.max_frames is not None:
            self.frames = self.frames[:self.max_frames]
    
    def run(self, file_id: str = None, dataset_name: str = None):
        '''运行视频生成
        
        input
        -----
        file_id: str, 文件ID，当path为None时使用
        dataset_name: str, 数据集名称，当path为None时使用
        '''
        # 视频文件名
        if self.path is not None:
            video_name = os.path.basename(self.path).split('.')[0] + '.mp4'
        else:
            # 当path为None时，使用数据集名称和文件ID生成文件名
            video_name = f"{dataset_name}_{file_id}.mp4"
        video_path = os.path.join(self.save_dir, video_name)
        
        # 创建视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, self.fps, (self.video_width, self.video_height))
        
        # 处理每一帧
        handle = tqdm(self.frames)
        for frame in handle:
            handle.set_description(f"Processing frame {frame}")
            # 获取当前帧的所有车辆数据
            frame_data = self.df[self.df[self.time_idx] == frame]
            
            # 创建空白图像
            img = np.ones((self.video_height, self.video_width, 3), dtype=np.uint8) * 255
            
            # 绘制每个车辆
            for _, row in frame_data.iterrows():
                # 缩放坐标到视频尺寸（反转y轴以符合标准平面坐标系）
                x = int((row[self.x_idx] - self.x_min) / self.x_range * (self.video_width - 100) + 50)
                y = int((self.video_height - 100) - (row[self.y_idx] - self.y_min) / self.y_range * (self.video_height - 100) + 50)
                
                # 计算速度对应的颜色
                speed = row['speed']
                color = self.cmap(self.norm(speed))[:3]  # 获取RGB颜色
                color = tuple(int(c * 255) for c in color[::-1])  # 转换为BGR格式
                
                # 绘制轨迹点（带黑色外轮廓）
                cv2.circle(img, (x, y), 8, (0, 0, 0), 2)  # 黑色外轮廓
                cv2.circle(img, (x, y), 6, color, -1)  # 速度颜色填充
                
                # 绘制车辆ID
                car_id = int(row[self.car_idx])
                cv2.putText(img, f"{car_id}", (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            # 添加帧号
            cv2.putText(img, f"Frame: {frame}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            
            # 写入视频
            out.write(img)
        
        # 释放视频编码器
        out.release()
        print(f"Video saved to: {video_path}")
        return video_path


def main_datasets(
    parquet_dir: str = r'E:\datasets\a微观轨迹数据集\parquet_original_value\train',
    output_dir: str = r'E:\datasets\a微观轨迹数据集\video_output',
    max_frames: int = None
):
    '''从parquet格式快速读取数据集并生成视频
    
    input
    -----
    parquet_dir: str, parquet文件目录
    output_dir: str, 视频输出目录
    max_frames: int, 最大处理帧数, 默认为None（不设上限）
    '''  
    from parquet_read import ParquetReader
    parquet_reader = ParquetReader(parquet_dir)
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 数据集
    datasets = ['NGSIM', 'highD', 'Mitra',  'ZEN', 'RAOYUE']
    # 统一的数据列配置（全部数据列为id frame x y vx vy ax ay width length lane）
    config = {
        'car_idx': 'id',
        'time_idx': 'frame',
        'x_idx': 'x',
        'y_idx': 'y',
        'vx_idx': 'vx',
        'vy_idx': 'vy'
    }
    # 遍历每个数据集
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        file_ids = parquet_reader.getFileIds(dataset_name)
        for file_id in file_ids:
            # 从parquet文件读取数据
            df = parquet_reader.load_dataset(
                dataset_name=dataset_name, 
                file_id=file_id
            )
            print(f'Processing file: {file_id}, Read data shape: {df.shape}')
            # 创建TrajVideoPlot实例并生成视频
            tvp = TrajVideoPlot(
                path=None,  # 不使用文件路径，直接使用df
                car_idx=config['car_idx'],
                time_idx=config['time_idx'],
                x_idx=config['x_idx'],
                y_idx=config['y_idx'],
                vx_idx=config['vx_idx'],
                vy_idx=config['vy_idx'],
                save_dir=output_dir,
                fps=10,
                video_width=1920,
                video_height=1080,
                max_frames=max_frames
            )
            
            # 直接使用读取的DataFrame
            tvp.df = df
            # 重新计算必要的属性
            tvp._recalculate_attributes()
            # 生成视频
            tvp.run(file_id=file_id, dataset_name=dataset_name)


if __name__ == '__main__':
    # 处理所有数据集（测试时设置max_frames=1000）
    main_datasets()
