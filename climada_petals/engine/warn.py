import numpy as np
from scipy import sparse
from scipy import ndimage
import skimage.morphology as morph
import copy
import cv2

class Warn:

    kernel_sizes = [2, 3, 7, 15]
    """
    kernel_sizes = [1, 3, 7, 15,
             1, 3, 7, 15,
             2, 3, 7, 15,
             1, 4, 8, 15]
    """
    def __init__(self, thresholds, data, ):
        self.thresholds = thresholds
        self.data = data

    def thresholding(self):
        m_thrs = np.zeros_like(self.data)
        for i in range(1, len(self.thresholds) - 1):
            m_thrs[self.data > self.thresholds[i]] = i
        return m_thrs.astype(int)

    def algo(self, m_thrs):
        max_warn_level = m_thrs.max(axis=(0, 1))
        if max_warn_level == 0:
            return m

        warn_reg = 0
        cnt = 0
        # iterate from highest level to lowest (0 not necessary, because rest is level 0)
        for i in range(max_warn_level, 0, -1):
            w_l = self.filtering(m_thrs, warn_reg, max_warn_level - i,
                                 self.kernel_sizes[0], self.kernel_sizes[1],
                                 self.kernel_sizes[2], self.kernel_sizes[3])
            # keep warn regions of higher levels by taking maximum
            warn_reg = np.maximum(warn_reg, w_l)
            cnt += 4
        return warn_reg


    def filtering(self, m_thrs, warn_reg, obs, di1, er, di2, md):
        level = np.where(np.bitwise_or(warn_reg, m_thrs >= m_thrs.max(axis=(0, 1)) - obs),
                         m_thrs.max(axis=(0, 1)) - obs, 0)

        lvl = morph.dilation(level, morph.disk(di1))
        lvl = morph.erosion(lvl, morph.disk(er))
        lvl = morph.dilation(lvl, morph.disk(di2))
        lvl = ndimage.median_filter(lvl, size=md)
        # lvl = morph.dilation(sa, morph.square(kernel_0))
        return lvl

    def remove_small_regions(self, warn_reg, size_thr):
        max_lvl = np.max(warn_reg, axis=(0, 1))
        min_lvl = np.min(warn_reg, axis=(0, 1))

        # increase levels of too small regions to max level of this warning
        num_labels, labels = cv2.connectedComponents(np.uint8(warn_reg))
        for i in range(num_labels):
            # use scipy label instead
            cnt = np.count_nonzero(labels == i)
            if cnt <= size_thr:
                warn_reg[labels == i] = max_lvl

        # level 0: set to small regions to max level
        level = copy.deepcopy(warn_reg)
        level[warn_reg == 0] = 1234
        level[level != 1234] = 0
        num_labels, labels = cv2.connectedComponents(np.uint8(level))
        for j in range(num_labels):
            cnt = np.count_nonzero(labels == j)
            if cnt <= size_thr:
                warn_reg[labels == j] = max_lvl

        # correct the max_lvl regions generated before down,
        # until the new regions with it are larg enough
        for i in range(max_lvl, min_lvl, -1):
            level = copy.deepcopy(warn_reg)
            level[warn_reg != i] = 0
            num_labels, labels = cv2.connectedComponents(np.uint8(level))
            for j in range(num_labels):
                cnt = np.count_nonzero(labels == j)
                if cnt <= size_thr:
                    warn_reg[labels == j] = i - 1

        return warn_reg