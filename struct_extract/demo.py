import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append('../')
import cv2
import os
from PIL import Image
from rog_smooth import rog_smooth

if __name__ == '__main__':
    files = os.listdir('input')
    for filename in files:
        I = np.array(Image.open(os.path.join('input', filename)), dtype=np.float) / 255
        if I.shape[-1] == 4:
            I = I[:, :, :-1]

        plt.figure()
        
        # plt.subplot(1, 3, 1)
        # plt.imshow(I)
        # plt.axis('off')
        # plt.title('Input')

        I1 = cv2.imread(os.path.join('input', filename), 0)
        edge1 = cv2.Canny(I1, 50, 150)
        # plt.subplot(1, 3, 2)
        # plt.imshow(edge1, cmap='gray')
        # plt.axis('off')
        # plt.title('edge from input')
        
        # RoG Smooth
        res = rog_smooth(I, 0.01, 2, 4, 4)
        res = np.clip(res, 0, 1)
        res = (res * 255).astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(res, 50, 150)

        # plt.subplot(1, 3, 3)
        # plt.imshow(edge, cmap='gray')
        # plt.axis('off')
        # plt.title('edge from rog')

        # plt.show()

        # save output
        # matplotlib.image.imsave(os.path.join('output_mine', filename), edge)
        cv2.imwrite(os.path.join('output', filename), edge)
