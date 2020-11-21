import numpy as np
import math
from scipy.signal import convolve2d
 

def compute_psnr(target, groundtruth):
    """
    计算图像target与原图像ref的psnr
    :param target: 进行细节增强后的图像，shape = [M,N,C] , C为通道数，M、N为图像竖向长度、横向长度，图像的值范围为[0,1]
    :param groundtruth: 原图像, shape = [M,N,C]
    :return: psnr
    """
    # target:目标图像  groundtruth:参考图像  scale:尺寸大小
    # assume RGB image
    target = np.array(target, dtype=np.double)
    groundtruth = np.array(groundtruth, dtype=np.double)
    diff = target - groundtruth
    MSE = np.mean(diff ** 2.0, dtype=np.double)
    # MSE_sum = np.sum(diff ** 2.0)
    # print("(diff ** 2.0).size",(diff ** 2.0).size)
    # print("MSE_sum",MSE_sum)
    # print("MSE",MSE)
    psnr = 10*math.log10(1.0/MSE)
    return psnr

def _matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def _filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def _compute_single_cannel_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    assert im1.shape == im2.shape, "两张图片的shape不等"
    assert len(im1.shape) <= 2, "Please input the images with 1 channel"

    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = _matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = _filter2(im1, window, 'valid')
    mu2 = _filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = _filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = _filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))


def compute_ssim(target, groundtruth):
    """
    计算图像target与原图像ref的ssim, 图片大小必须大于11*11
    :param target: 进行细节增强后的图像，shape = [M,N,C] , C为通道数，M、N为图像竖向长度、横向长度，图像的值范围为[0,1]
    :param groundtruth: 原图像, shape = [M,N,C]
    :return: ssim
    """

    target = np.array(target, dtype=np.double)
    groundtruth = np.array(groundtruth, dtype=np.double)

    assert target.shape == groundtruth.shape, "两张图片的shape不等"

    M, N, C = target.shape
    assert M >= 11 and N >= 11, "图片大小必须大于11*11"
    total_ssim = 0

    for c in range(C):
        _target = target[:,:,c]
        _groundtruth = groundtruth[:,:,c]
        ssim = _compute_single_cannel_ssim(_target, _groundtruth, L=1.0)
        total_ssim += ssim
    
    return total_ssim/C

