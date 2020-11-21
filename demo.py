import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import os
from PIL import Image
from rog_smooth import rog_smooth


if __name__ == '__main__':
    img = Image.open("pics/fish.png")
    # 转为double类型并归一化到[0,1]
    img = np.array(img, dtype=np.float) / 255
    if img.shape[-1] == 4:
        img = img[:, :, :-1]

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('input')

    start = time.time()
    img = rog_smooth(img, 0.01, 10, 15, 5)
    end = time.time()
    print(f'total time:{end - start}')

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title('rog smooth')

    plt.show()

    matplotlib.image.imsave("pics/fish_rog2.png", img)
