'''
    图片转npz格式
'''

import os

# pic =cv2.imread(r'C:\Users\DELL\Desktop\7.30\detection\labels\1.png')
# print(pic)
import glob
import cv2
import numpy as np


def npz():
    # 图像路径
    path = r"E:\Result\BSIRST_v2 Dataset\images"
    # 项目中存放训练所用的npz文件路径
    path2 = r"E:\Result\BSIRST_v2 Dataset\V2_npz\\"

    for i, img_path in enumerate(glob.glob(os.path.join(path, '*.png'))):
        # 读入图像
        name = os.path.split(img_path)[-1].split('.')[0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读入标签
        label_path = img_path.replace('images', 'mask1')
        label = cv2.imread(label_path, flags=0)

        # 保存npz
        # np.savez(path2 + name, image=image, label=label, edge=edge)
        np.savez(path2 + name, image=image, label=label)
        print('------------', name)

    print('ok')


if __name__ == "__main__":
    npz()
