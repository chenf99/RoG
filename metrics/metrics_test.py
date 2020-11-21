import metrics
import numpy as np

from PIL import Image


ref_image_path = "E:/中山大学/研一上/图像处理/辣鸡大作业/DeFilter-master/DeFilter-master/filter_code/imgs_tsmooth/3.png"
target_image_path = "E:/中山大学/研一上/图像处理/辣鸡大作业/DeFilter-master/DeFilter-master/filter_code/imgs_tsmooth/3_reversed_result.png"

ref=np.array(Image.open(ref_image_path))/255.0
target=np.array(Image.open(target_image_path))/255.0

# 去除alpha通道
ref=ref[:,:,:3]
target=target[:,:,:3]



def compute_psnr_test():
    # psnr = metrics.compute_psnr(target, ref)
    psnr = metrics.compute_psnr(target, ref)
    print("psnr",psnr)

def compute_ssim_test():
    # psnr = metrics.compute_psnr(target, ref)
    ssim = metrics.compute_ssim(target, ref)
    print("ssim",ssim)


if __name__ == '__main__':
    # compute_psnr_test()
    compute_ssim_test()
    pass