import os
from PIL import Image


def combine_images(dir: str, output_name: str = 'combined.jpg', direction: str = 'v'):
    '''将文件夹下的所有图片合并成一张图片

    图片的排列顺序为从上到下

    input
    -----
    dir: str, 图片所在文件夹路径
    output_name: str, 输出图片的文件名
    direction: str, 排列方向, 可选v, h. v表示从上到下, h表示从左到右.
    '''
    # 获取文件夹下的所有图片
    images = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.jpg') and 'combine' not in f]
    images.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    # 读取第一张图片, 获取图片的宽度和高度
    img = Image.open(images[0])
    width, height = img.size
    # 创建一个新的空白图片, 大小为所有图片的宽度和高度之和
    if direction == 'v':
        result = Image.new(img.mode, (width, height * len(images)))
    elif direction == 'h':
        result = Image.new(img.mode, (width * len(images), height))
    # 将所有图片粘贴到新的图片中
    for i, image in enumerate(images):
        img = Image.open(image)
        if direction == 'v':
            result.paste(img, (0, i * height))
        elif direction == 'h':
            result.paste(img, (i * width, 0))
    # 保存新的图片
    save_path = os.path.join(dir, output_name)
    result.save(save_path)
    return save_path


if __name__ == '__main__':
    combine_images(r'D:\myscripts\pro\output\model0\trajectory', direction='v')
