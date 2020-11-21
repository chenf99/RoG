import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
sys.path.append('../')
from rog_smooth import rog_smooth
from PIL import Image
from detail_enhancement import detailenhance


if __name__ == '__main__':
    files = os.listdir('input_mine')
    for filename in files:
        I = np.array(Image.open(os.path.join('input_mine', filename)), dtype=np.float) / 255
        if I.shape[-1] == 4:
            I = I[:, :, :-1]

        plt.figure()

        # plt.subplot(2, 2, 1)
        # plt.imshow(I)
        # plt.axis('off')
        # plt.title('Input')
        
        # RoG Smooth
        I0 = rog_smooth(I, 0.001, 0.5, 1.0, 1)
        I1 = rog_smooth(I, 0.001, 1.0, 1.5, 1)
        # Detail Enhancement
        coarse = detailenhance(I, I0, I1, 1, 25)
        # plt.subplot(2, 2, 2)
        # plt.imshow(coarse)
        # plt.axis('off')
        # plt.title('Coarse-scale boost')

        fine = detailenhance(I, I0, I1, 12, 1)
        # plt.subplot(2, 2, 3)
        # plt.imshow(fine)
        # plt.axis('off')
        # plt.title('Fine-scale boost')

        combine = (fine + coarse) / 2
        # plt.subplot(2, 2, 4)
        # plt.imshow(combine)
        # plt.axis('off')
        # plt.title('Combine')

        # plt.show()

        # save output
        matplotlib.image.imsave(os.path.join('output2', filename), np.clip(combine, 0, 1))
