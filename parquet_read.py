import warnings
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')  # 忽略无关警告


class ParquetReader:
    """
    Parquet 数据读取类
    核心功能：从 Parquet 文件中读取和筛选数据，支持分批加载
    """
    def __init__(
        self,
        parquet_dir: Union[str, Path],
    ):
        """
        初始化读取器
        :param PARQUET_DIR: Parquet文件目录路径
        """
        self.parquet_dir = Path(parquet_dir)

        # 验证目录存在
        if not self.parquet_dir.exists():
            raise FileNotFoundError(f"Parquet目录不存在: {self.parquet_dir}")

    def load_filtered_data(
        self,
        filters: Optional[List] = None,
        columns: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        dataset: Optional[str] = None,
        file_id: Optional[Union[str, List[str]]] = None,
        id: Optional[Union[int, List[int]]] = None,
        datetime_start: Optional[str] = None,
        datetime_end: Optional[str] = None
    ) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        """
        按需加载筛选后的Parquet数据（支持分批加载）
        :param filters: 筛选条件，如 [("id", "in", [0]), ("file_id", "=", "track1")]
        :param columns: 需要加载的列（None表示加载所有列）
        :param batch_size: 分批加载的批次大小（None表示一次性加载）
        :param dataset: 筛选指定的数据集名称
        :param file_id: 筛选指定的文件标识（单个或多个）
        :param id: 筛选指定的车辆ID（单个或多个）
        :param datetime_start: 筛选指定的开始时间
        :param datetime_end: 筛选指定的结束时间
        :return: 筛选后的DataFrame或分批迭代器
        """
        try:
            # 构建筛选条件
            final_filters = []
            
            # 添加用户提供的筛选条件
            if filters:
                final_filters.extend(filters)
            
            # 添加dataset筛选条件
            if dataset:
                final_filters.append(("dataset", "=", dataset))
            
            # 添加file_id筛选条件
            if file_id:
                if isinstance(file_id, list):
                    final_filters.append(("file_id", "in", file_id))
                else:
                    final_filters.append(("file_id", "=", file_id))
            
            # 添加id筛选条件
            if id is not None:
                if isinstance(id, list):
                    final_filters.append(("id", "in", id))
                else:
                    final_filters.append(("id", "=", id))
            
            # 添加datetime筛选条件
            if datetime_start:
                final_filters.append(("datetime", ">=", datetime_start))
            if datetime_end:
                final_filters.append(("datetime", "<=", datetime_end))
            
            # 如果没有筛选条件，设置为None
            if not final_filters:
                final_filters = None
            
            # 使用pyarrow直接读取
            if batch_size:
                # 分批加载（适合超大数据集）
                class PyArrowBatchReader:
                    def __init__(self, dataset, batch_size):
                        self.dataset = dataset
                        self.batch_size = batch_size
                        self.scanner = dataset.scanner(batch_size=batch_size)
                        
                    def __iter__(self):
                        return self
                        
                    def __next__(self):
                        batch = self.scanner.to_batches().next()
                        if batch is None:
                            raise StopIteration
                        return batch.to_pandas()
                
                # 创建pyarrow数据集
                dataset = pq.ParquetDataset(
                    str(self.parquet_dir),
                    filters=final_filters
                )
                
                return PyArrowBatchReader(dataset, batch_size)
            else:
                # 一次性加载
                # 创建pyarrow数据集
                dataset = pq.ParquetDataset(
                    str(self.parquet_dir),  
                    filters=final_filters
                )
                
                # 读取所有数据
                table = dataset.read(columns=columns)
                df = table.to_pandas()
                return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            return pd.DataFrame()  # 返回空DataFrame避免程序崩溃

    def load_dataset(
        self,
        dataset_name: str,
        file_id: Optional[str] = None,
        columns: Optional[List[str]] = None,
        random_ratio: Optional[float] = None,
        random_num: Optional[int] = None,
        ) -> pd.DataFrame:
        """
        加载指定数据集的所有数据
        :param dataset_name: 数据集名称
        :param file_id: 筛选指定的文件标识
        :param columns: 需要加载的列（None表示加载所有列）
        :param random_ratio: 随机采样比例（None表示不采采样）
        :param random_num: 随机采样数量（None表示不采随机采样数据）
        :return: 数据集的所有DataFrame或torch张量
        """
        if random_ratio is not None and random_num is not None:
            raise ValueError("random_ratio and random_num cannot be set at the same time")
        filters=[("dataset", "=", dataset_name)]
        if file_id:
            filters.append(("file_id", "=", file_id))
        
        dataset = pq.ParquetDataset(str(self.parquet_dir),filters=filters)
        table = dataset.read(columns=columns)
        df = table.to_pandas()

        # 随机采样
        if random_ratio is not None:
            data_num = len(df)
            sample_size = min(int(data_num * random_ratio), data_num)
            # 随机选择数据
            df = df.sample(n=sample_size, random_state=1234)
        elif random_num is not None:
            # 随机选择数据
            random_num = min(random_num, len(df))
            df = df.sample(n=random_num, random_state=1234)
        # 重置index
        return df.reset_index(drop=True)

    def getFileIds(self, dataset_name: str) -> List[str]:
        """
        获取指定数据集的所有文件标识
        :param dataset_name: 数据集名称
        :return: 数据集的所有文件标识列表
        """
        # 遍历对应parquet文件夹下面的文件夹名称
        parquet_dir = self.parquet_dir / f'dataset={dataset_name}'
        file_ids = [file_id.name.replace('file_id=', '') for file_id in parquet_dir.iterdir() if file_id.is_dir()]
        file_ids.sort()
        return file_ids

def checkNan(df: pd.DataFrame) -> bool:
    """
    检查DataFrame中是否存在NaN值
    :param df: 输入DataFrame
    :return: 如果存在NaN值则返回True，否则返回False
    """

    res = df.isnull().sum()
    return res

def checkLoadDatasetTime():
    dataset_name = "highD"
    dataset = pq.ParquetDataset(
                    r'E:\datasets\a微观轨迹数据集\parquet\train\dataset=highD',
                )

    # 依次加载出每个file_id下parquet文件的单行的数据，检查加载时间
    import time
    import os
    import torch
    start_time = time.time()
    total_lines = 0
    base_total_time = 0
    
    # 初始化每行的总时间
    line1_total_time = 0  # 创建tmp_dataset
    line2_total_time = 0  # 读取table
    line3_total_time = 0  # 转为DataFrame
    line4_total_time = 0  # 转为torch张量
    
    for i, file in enumerate(dataset.files):
        if i >= 10:
            break
        print(f'reading file {file}')
        file_id = os.path.dirname(file).split('/')[-1].replace('file_id=', '')

        parquet_file = pq.ParquetFile(file)
        num_rows = parquet_file.metadata.num_rows
        total_lines += num_rows
        file_start_time = time.time()
        for row_id in range(num_rows):
            row_id += 1
            # 加载出file对应的row_id列数值为row_id的那一行数据
            # 第1行：创建tmp_dataset
            line1_start = time.time()
            tmp_dataset = pq.ParquetDataset(
                    str(PARQUET_DIR),
                    filters=[("file_id", "=", file_id), ("row_id", "=", row_id)]
                )
            line1_end = time.time()
            line1_total_time += (line1_end - line1_start)
            
            # 第2行：读取table
            line2_start = time.time()
            table = tmp_dataset.read()
            line2_end = time.time()
            line2_total_time += (line2_end - line2_start)
            
            # 转为torch张量
            # 第3行：转为DataFrame
            line3_start = time.time()
            df = table.to_pandas()
            line3_end = time.time()
            line3_total_time += (line3_end - line3_start)
            
            # 第4行：转为torch张量
            line4_start = time.time()
            tensor = torch.from_numpy(df.values[:, :-2].astype(float))
            line4_end = time.time()
            line4_total_time += (line4_end - line4_start)
        file_end_time = time.time()
        file_load_time = file_end_time - file_start_time
        base_total_time += file_load_time
        print(f"file {file_id} loading time: {file_load_time:.2f}秒, num_rows: {num_rows}, ave: {file_load_time / num_rows:.2f}秒")

    end_time = time.time()
    total_dataset_load_time = end_time - start_time
    print(f"加载时间: {total_dataset_load_time:.2f}秒, 总行数：{total_lines}, 每行平均加载时间: {total_dataset_load_time / total_lines:.2f}秒")
    print(f'其中正式文件的加载时间: {base_total_time:.2f}秒, 额外多花时间: {total_dataset_load_time - base_total_time:.2f}秒, 时间占比: {base_total_time / total_dataset_load_time:.2f}')
    
    # 输出每行的总耗时和平均耗时
    print("\n每行代码执行时间统计:")
    print(f"第1行 (创建tmp_dataset): 总耗时 {line1_total_time:.4f}秒, 平均耗时 {line1_total_time / total_lines:.6f}秒")
    print(f"第2行 (读取table): 总耗时 {line2_total_time:.4f}秒, 平均耗时 {line2_total_time / total_lines:.6f}秒")
    print(f"第3行 (转为DataFrame): 总耗时 {line3_total_time:.4f}秒, 平均耗时 {line3_total_time / total_lines:.6f}秒")
    print(f"第4行 (转为torch张量): 总耗时 {line4_total_time:.4f}秒, 平均耗时 {line4_total_time / total_lines:.6f}秒")


def read_parquet_with_pandas(file_path: Union[str, Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    使用 pandas 读取单个 parquet 文件
    :param file_path: parquet 文件路径
    :param columns: 需要加载的列（None 表示加载所有列）
    :return: 读取后的 DataFrame
    """
    try:
        df = pd.read_parquet(file_path, columns=columns)
        return df
    except Exception as e:
        print(f"使用 pandas 读取 parquet 文件失败: {e}")
        return pd.DataFrame()  # 返回空 DataFrame 避免程序崩溃


# ------------------------------ 脚本使用示例 ------------------------------
if __name__ == "__main__":
    # checkLoadDatasetTime()

    # reader = ParquetReader()
    # # 测试读取数据集并查看空值数量
    # dataset_name = 'RAOYUE'
    # df = reader.load_dataset(dataset_name)
    # print(len(df))
    # df = reader.load_dataset(dataset_name, random_ratio=0.1)
    # print(len(df))

    reader = ParquetReader(r'E:/graduation/prediction/20260209_005714')
    # 测试读取数据集并查看空值数量
    dataset_name = 'RAOYUE'
    df = reader.load_dataset(dataset_name)
    print(len(df))

    # # 示例1：按dataset筛选
    # print("\n1. 按dataset筛选:")
    # for ds_name in ["NGSIM"]:     # ["Mitra", "ZEN", 'highD', "NGSIM", "RAOYUE"]
    #     print(f"筛选数据集: {ds_name}")
    #     filtered_data1 = reader.load_filtered_data(
    #         dataset=ds_name,
    #         columns=['x', 'y', 'vx', 'vy']
    #     )
    #     print(f"筛选后数据量: {len(filtered_data1)} 条")

    # ds_name = "Mitra"
    # # 示例2：按file_id筛选
    # print("\n2. 按file_id筛选:")
    # filtered_data2 = reader.load_filtered_data(
    #     dataset=ds_name,
    #     file_id="1",
    #     columns=['x', 'y', 'vx', 'vy']
    # )
    # print(f"筛选后数据量: {len(filtered_data2)} 条")
    
    # # 示例3：按id筛选
    # print("\n3. 按id筛选:")
    # filtered_data3 = reader.load_filtered_data(
    #     dataset=ds_name,
    #     file_id="1",
    #     id=1,
    #     columns=['x', 'y', 'vx', 'vy']
    # )
    # print(f"筛选后数据量: {len(filtered_data3)} 条")

    # # 打印前几行数据查看结果
    # if not filtered_data3.empty:
    #     print("\n数据预览:")
    #     print(filtered_data3.head())

    # # 分批加载（适合大模型训练）
    # print("\n5. 分批加载示例:")
    # batch_reader = reader.load_filtered_data(
    #     dataset="highd",
    #     id=1,
    #     columns=['id', 'x', 'y', 'vx', 'vy', 'dataset', 'file_id', 'datetime'],
    #     batch_size=10000
    # )
    # for batch_idx, batch_data in enumerate(batch_reader):
    #     print(f"处理第 {batch_idx+1} 批数据，数据量: {len(batch_data)} 条")
    #     # 这里可以将batch_data传入大模型训练
    #     # model.train(batch_data)
