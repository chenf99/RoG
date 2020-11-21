import numpy as np
import scipy
from scipy import signal
from scipy import sparse
import cv2


def rog_smooth(img, lamb=0.01, sigma1=1, sigma2=3, K=1, dec=2.0, sep=False):
    out = img.copy()
    for _ in range(K):
        wx, wy = computeReWeights(out, sigma1, sigma2, sep)
        out = solveLinearEquation(img, wx, wy, lamb)
        sigma1 /= dec
        sigma2 /= dec
    return out


def computeReWeights(img, sigma1, sigma2, sep):
    eps = 0.00001
    
    dx = np.diff(img, 1, 1)
    dx = np.pad(dx, ((0, 0), (0, 1), (0, 0)))
    dy = np.diff(img, 1, 0)
    dy = np.pad(dy, ((0, 1), (0, 0), (0, 0)))

    if sep == True:
        gdx1 = fastBlur(dx, sigma1)
        gdy1 = fastBlur(dy, sigma1)
    else:
        do = np.sqrt(dx ** 2 + dy ** 2)
        gdo = fastBlur(do, sigma1)
        gdx1 = gdo
        gdy1 = gdo
    
    gdx2 = fastBlur(dx, sigma2)
    gdy2 = fastBlur(dy, sigma2)

    wx = 1.0 / np.maximum(np.mean(np.abs(gdx1), 2) * np.mean(np.abs(gdx2), 2), eps)
    wy = 1.0 / np.maximum(np.mean(np.abs(gdy1), 2) * np.mean(np.abs(gdy2), 2), eps)
    wx = fastBlur(wx, sigma1 / 2)
    wy = fastBlur(wy, sigma1 / 2)
    wx[:, -1] = 0
    wy[-1, :] = 0

    # wx: [H, W]
    # wy: [H, W]
    return wx, wy


def fastBlur(img, sigma):
    k_size = round(5 * sigma) | 1
    g = getGaussianKernel((1, k_size), sigma)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    res = img
    for c in range(res.shape[-1]):
        ret = signal.convolve2d(img[:, :, c], g, 'same')
        ret = signal.convolve2d(ret, g.T, 'same')
        res[:, : , c] = ret
    return res


def getGaussianKernel(kernel_size, sigma):
    kx = cv2.getGaussianKernel(kernel_size[0], sigma)
    ky = cv2.getGaussianKernel(kernel_size[1], sigma)
    return np.multiply(kx, ky.T)


def solveLinearEquation(img, wx, wy, lamb):
    h, w, c = img.shape
    n = h * w

    wx = np.reshape(wx, (-1, 1), 'F')
    wy = np.reshape(wy, (-1, 1), 'F')

    ux = np.pad(wx, ((h, 0), (0, 0)))
    ux = ux[:-h]
    uy = np.pad(wy, ((1, 0), (0, 0)))
    uy = uy[:-1]

    D = wx + ux + wy + uy

    B = sparse.spdiags(np.concatenate((-wx, -wy), axis=1).T, [-h, -1], n, n)
    L = B + B.T + sparse.spdiags(D.T, 0, n, n)

    A = sparse.eye(n) + lamb * L

    output = img.copy()
    # for i in range(c):
    #     tin = img[:, :, i]
    #     tout = sparse.linalg.spsolve(A, tin.flatten('F'))
    #     output[:, :, i] = np.reshape(tout, (h, w), 'F')

    # speed up by PCG
    for i in range(c):
        tin = img[:, :, i]
        tout, _ = sparse.linalg.cg(A, tin.flatten('F'), tol=0.0000001, maxiter=200)
        output[:, :, i] = np.reshape(tout, (h, w), 'F')
    return output
