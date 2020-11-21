import numpy as np


def detailenhance(L, L0, L1, w0, w1):
    detail0 = L - L0
    detail0 = sigmoid(detail0, w0)
    detail1 = L0 - L1
    detail1 = sigmoid(detail1, w1)
    base = L1
    res = base + detail1 + detail0
    res = sigmoid(res, 1)
    return res


def sigmoid(x, a):
    y = 1.0 / (1 + np.exp(-a * x)) - 0.5

    y_05 = 1.0 / (1 + np.exp(-a * 0.5)) - 0.5
    y = y * (0.5 / y_05)

    return y
