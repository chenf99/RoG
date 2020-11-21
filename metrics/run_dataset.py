import metrics
import numpy as np
import csv
from PIL import Image

baselines = ['LLF','RTV','WLS']

def main():
    psnr_rows = []
    ssim_rows = []
    for i in range(500):
        print(i)
        psnr_list = []
        ssim_list = []
        for baseline in baselines:
            ref_image_path = "./images/%d.png" % i
            target_image_path = "./images_%s/%d.png" % (baseline, i)
            ref=np.array(Image.open(ref_image_path))/255.0
            target=np.array(Image.open(target_image_path))/255.0

            # 去除alpha通道
            ref=ref[:,:,:3]
            target=target[:,:,:3]

            psnr = metrics.compute_psnr(target, ref)
            ssim = metrics.compute_ssim(target, ref)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
        psnr_rows.append(psnr_list)
        ssim_rows.append(ssim_list)
      
    headers = baselines

    with open('evaluation/psnr.csv','w',newline='')as f:
        f_csv = csv.writer(f,delimiter=",")
        f_csv.writerow(headers)
        f_csv.writerows(psnr_rows)
    
    with open('evaluation/ssim.csv','w',newline='')as f:
        f_csv = csv.writer(f,delimiter=",")
        f_csv.writerow(headers)
        f_csv.writerows(ssim_rows)


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
    # compute_ssim_test()
    main()
    pass