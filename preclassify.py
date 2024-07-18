# @ Author: Feng Gao, Ocean University of China
# @ Date: 2020-02-22
# @ Email: gaofeng@ouc.edu.cn
# @ Homepage: http://feng-gao.cn

import numpy as np
from scipy import signal
from scipy.linalg import norm
from scipy.spatial.distance import cdist


def del2(im):
    [ylen, xlen] = im.shape
    im_new = np.zeros([ylen, xlen], dtype=np.float32)
    for j in range(1, ylen - 1):
        for i in range(1, xlen - 1):
            im_new[j, i] = (im[j - 1, i] + im[j + 1, i] + im[j, i - 1] + im[j, i + 1]) / 4 - im[j, i]
    return im_new


def srad(im, delta):
    q0 = 1
    for n in range(1, 6):
        [ylen, xlen] = im.shape
        X = np.zeros([ylen + 2, xlen + 2], dtype=np.float32)
        X[1:ylen + 1, 1:xlen + 1] = im
        # padding
        X[0, 1:xlen + 1] = im[0, :]
        X[ylen + 1, 1:xlen + 1] = im[ylen - 1, :]
        X[:, 0] = X[:, 1]
        X[:, xlen + 1] = X[:, xlen]

        q0 = q0 * np.exp(-delta)
        gRx = signal.convolve2d(X, [[0, 0, 0], [0, 1, -1], [0, 0, 0]], mode='same', boundary='symm')
        gRy = signal.convolve2d(X, [[0, -1, 0], [0, 1, 0], [0, 0, 0]], mode='same', boundary='symm')
        gLx = signal.convolve2d(X, [[0, 0, 0], [1, -1, 0], [0, 0, 0]], mode='same', boundary='symm')
        gLy = signal.convolve2d(X, [[0, 0, 0], [0, -1, 0], [0, 1, 0]], mode='same', boundary='symm')
        q1 = np.sqrt(gRx * gRx + gRy * gRy + gLx * gLx + gLy * gLy) / (X + 0.0001)
        q2 = 4 * del2(X) / (X + 0.0001)
        q = np.sqrt((1 / 2 * (q1 * q1) - 1 / 16 * (q2 * q2)) / ((1 + 1 / 4 * q2) * (1 + 1 / 4 * q2) + 0.01))
        c = 1 / (1 + ((q * q - q0 * q0) / (q0 * q0 * (1 + q0 * q0))))
        d = signal.convolve2d(c, [[0, 0, 0], [0, 0, -1], [0, 0, 0]], mode='same', boundary='symm') * \
            signal.convolve2d(X, [[0, 0, 0], [0, 1, -1], [0, 0, 0]], mode='same', boundary='symm') + \
            signal.convolve2d(c, [[0, 0, 0], [0, -1, 0], [0, 0, 0]], mode='same', boundary='symm') * \
            signal.convolve2d(X, [[0, 0, 0], [-1, 1, 0], [0, 0, 0]], mode='same', boundary='symm') + \
            signal.convolve2d(c, [[0, -1, 0], [0, 0, 0], [0, 0, 0]], mode='same', boundary='symm') * \
            signal.convolve2d(X, [[0, -1, 0], [0, 1, 0], [0, 0, 0]], mode='same', boundary='symm') + \
            signal.convolve2d(c, [[0, 0, 0], [0, -1, 0], [0, 0, 0]], mode='same', boundary='symm') * \
            signal.convolve2d(X, [[0, 0, 0], [0, 1, 0], [0, -1, 0]], mode='same', boundary='symm')
        X = X + delta / 4 * d
        im = X[1:ylen + 1, 1:xlen + 1]
    return im


def dicomp(im1, im2):
    im1 = srad(im1, 0.15)
    im2 = srad(im2, 0.15)
    im_di = abs(np.log((im1 + 1) / (im2 + 1)))
    im_di = srad(im_di, 0.15)
    return im_di
    # im1-im2


# hiearchical FCM clustering
# in the preclassification map,
# pixels with high probability to be unchanged are labeled 1
# pixels with high probability to be changed are labeled 2
# pixels with uncertainty are labeled 1.5

def hcluster(pix_vec, im_di):
    print('... ... 1st round clustering ... ...')
    fcm = FCM(n_clusters=2)
    fcm.fit(pix_vec)
    fcm_lab = fcm.u.argmax(axis=1)

    # 变化类像素数目的上下界
    if sum(fcm_lab == 0) < sum(fcm_lab == 1):
        ttr = round(sum(fcm_lab == 0) * 1.25)
        ttl = round(sum(fcm_lab == 0) / 1.10)
    else:
        ttr = round(sum(fcm_lab == 1) * 1.25)
        ttl = round(sum(fcm_lab == 1) / 1.10)

    print('... ... 2nd round clustering ... ...')
    fcm = FCM(n_clusters=5)
    fcm.fit(pix_vec)
    fcm_lab = fcm.u.argmax(axis=1)
    ylen, xlen = im_di.shape
    idx = []
    idx_tmp = []
    idxmean = []
    res_lab = np.zeros(ylen * xlen, dtype=np.float32)
    for i in range(0, 5):
        idx_tmp.append(np.argwhere(fcm_lab == i))
        idxmean.append(im_di.reshape(ylen * xlen, 1)[idx_tmp[i]].mean())

    idx_sort = np.argsort(idxmean)
    for i in range(0, 5):
        idx.append(idx_tmp[idx_sort[i]])
    c = len(idx[4])
    res_lab[idx[4]] = 2
    flag_mid = 0
    for i in range(1, 5):
        c = c + len(idx[4 - i])
        if c < ttl:
            res_lab[idx[4 - i]] = 2
        elif c >= ttl and c < ttr:
            res_lab[idx[4 - i]] = 1.5
            flag_mid = 1
        elif flag_mid == 0:
            res_lab[idx[4 - i]] = 1.5
            flag_mid = 1
        else:
            res_lab[idx[4 - i]] = 1
    res_lab = res_lab.reshape(ylen, xlen)
    return res_lab


class FCM:
    def __init__(self, n_clusters=10, max_iter=150, m=2, error=1e-5, random_state=42):
        self.u, self.centers = None, None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.random_state = random_state

    def fit(self, X):
        N = X.shape[0]
        C = self.n_clusters
        centers = []
        r = np.random.RandomState(self.random_state)
        u = r.rand(N, C)
        u = u / np.tile(u.sum(axis=1)[np.newaxis].T, C)
        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()
            centers = self.next_centers(X, u)
            u = self.next_u(X, centers)
            iteration += 1
            # Stopping rule
            if norm(u - u2) < self.error:
                break
        self.u = u
        self.centers = centers
        return self

    def next_centers(self, X, u):
        um = u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).T

    def next_u(self, X, centers):
        return self._predict(X, centers)

    def _predict(self, X, centers):
        power = float(2 / (self.m - 1))
        temp = cdist(X, centers) ** power
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_
        return 1 / denominator_.sum(2)

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        u = self._predict(X, self.centers)
        return np.argmax(u, axis=-1)
