import os
import pandas as pd
import matplotlib.pyplot as plt


'''画出xy平面上的轨迹数据散点'''


FIGSIZE = (4.5, 6)
# plt字体设置为times new roman
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
# # plt横纵坐标标题字体大小
plt.rcParams['axes.labelsize'] = 14
# 全局字体默认大小
plt.rcParams['font.size'] = 13
# legend字体大小
plt.rcParams['legend.fontsize'] = 12


def trajXYScatter(csvPath: str,
                  xIndex: int, yIndex: int, laneIndex: int,
                  savePath: str = None):
    '''function trajXYScatter

    input
    -----
    csvPath: str, 轨迹数据路径
    xIndex: int, x坐标索引(csv文件中的列索引)
    yIndex: int, y坐标索引(csv文件中的列索引)
    laneIndex: int, 车道索引(csv文件中的列索引)
    savePath: str, 保存路径

    画出xy平面上的轨迹数据散点。
    '''
    # 指定数据和列
    data = pd.read_csv(csvPath)
    # data = data[data['lane'] > 0]       # for field data
    colX, colY = data.columns[xIndex], data.columns[yIndex]
    colLane = data.columns[laneIndex]

    # 画图
    plt.figure(figsize=FIGSIZE)
    plt.tight_layout()
    plt.scatter([0], [0], s=100, c="red", marker="o")  # 标注雷达原点
    # plt.scatter(data[data.columns[xIndex]], data[data.columns[yIndex]], s=1)
    for group, dfLane in data.groupby(colLane):
        # 加alpha会变糊
        plt.scatter(dfLane[colX], dfLane[colY],
                    label='lane'+str(group), s=1, alpha=0.5)

    # 添加元素
    if savePath is None:
        savePath = csvPath.split('.')[0] + '.png'
    saveFileName = os.path.basename(savePath).split('.')[0]
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    # plt.title(saveFileName)
    plt.legend(handletextpad=0.05, labelspacing=0.1, loc="best")
    # plt.legend(loc="best")
    plt.savefig(savePath, dpi=300)
    # plt.show()


if __name__ == '__main__':
    # 最初的result
    # xIndex = 1
    # yIndex = 2
    # laneIndex = 13
    #
    # field data
    xIndex = 3
    yIndex = 4
    laneIndex = 2

    # # 画单个文件
    csvPath = r"D:\myscripts\spill-detection\data\extractedData\2024-3-27-17_byDevice\K78+760_1.csv"
    trajXYScatter(csvPath, xIndex, yIndex, laneIndex)
