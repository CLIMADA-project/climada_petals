import logging
import numpy as np
import skimage.morphology as morph
import skimage.filters as filters
from skimage.measure import label
import copy
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass
class FilteringOrder:

    def __init__(self, thresholds, expand, operations, sizes):
        self.thresholds = thresholds
        self.expand = expand
        if len(operations) != len(sizes):
            LOGGER.warning('For every operation a filter size is needed and the other way around. '
                           'Please input the same number of operations and filter sizes.')
        self.operations = operations
        self.sizes = sizes


class Warn:
    """Explanation of defaults (also maybe transfer to config, talk to Emanuel)"""
    operations = ['DILATION', 'EROSION', 'DILATION', 'MEDIANFILTERING']
    sizes = [2, 3, 7, 15]

    def __init__(self, filter_data, data):
        self.filter_data = filter_data
        self.nr_thresholds = len(self.filter_data.thresholds) - 1
        self.warning = np.zeros_like(data)

    @classmethod
    def from_np_array(cls, data, thresholds, expand=True, operations=operations, sizes=sizes):  # pass defaults
        filter_data = FilteringOrder(thresholds, expand, operations, sizes)
        warn = cls(filter_data, data)
        d_thrs = warn.threshold_data(data)
        warn.filter_algorithm(d_thrs)
        return warn

    def threshold_data(self, data):
        if np.max(data) > np.max(self.filter_data.thresholds) or np.min(data) < np.min(self.filter_data.thresholds):
            LOGGER.warning('Values of data array are smaller/larger than defined thresholds. '
                           'Please redefine thresholds.')
        m_thrs = np.zeros_like(data)
        for i in range(1, len(self.filter_data.thresholds)):
            m_thrs[data > self.filter_data.thresholds[i]] = i
        return m_thrs.astype(int)

    def filter_algorithm(self, d_thrs):
        def filtering(d_thrs, warn_reg, curr_lvl):
            if self.filter_data.expand:  # select points where level is >= level under observation -> expands regions
                pts_curr_lvl = np.bitwise_or(warn_reg, d_thrs >= curr_lvl)
            else:  # select points where level is == level under observation -> not expanding regions
                pts_curr_lvl = d_thrs == curr_lvl
            reg_curr_lvl = np.where(pts_curr_lvl, curr_lvl, 0)  # set everything but pts of current level to 0

            for i in range(len(self.filter_data.operations)):
                if self.filter_data.operations[i] == 'DILATION':
                    reg_curr_lvl = morph.dilation(reg_curr_lvl, morph.disk(self.filter_data.sizes[i]))
                elif self.filter_data.operations[i] == 'EROSION':
                    reg_curr_lvl = morph.erosion(reg_curr_lvl, morph.disk(self.filter_data.sizes[i]))
                elif self.filter_data.operations[i] == 'MEDIANFILTERING':
                    filter = np.ones((self.filter_data.sizes[i], self.filter_data.sizes[i]))
                    reg_curr_lvl = filters.median(reg_curr_lvl, filter)
                else:
                    LOGGER.warning("The operation is not defined. "
                                   "Please select 'EROSION', 'DILATION', or 'MEDIANFILTERING'.")
            return reg_curr_lvl

        max_warn_level = np.max(d_thrs)
        if max_warn_level == 0:
            return np.zeros(d_thrs.shape)

        warn_reg = 0
        cnt = 0
        # iterate from highest level to lowest (0 not necessary, because rest is level 0)
        for i in range(max_warn_level, 0, -1):
            w_l = filtering(d_thrs, warn_reg, i)
            # keep warn regions of higher levels by taking maximum
            warn_reg = np.maximum(warn_reg, w_l)
            cnt += 4
        self.warning = warn_reg

    @staticmethod
    def remove_small_regions(warning, size_thr):

        def increase_levels(warning, size_thr):
            # increase levels of too small regions to max level of this warning
            labels = label(warning)
            for i in range(np.max(labels)):
                cnt = np.count_nonzero(labels == i)
                if cnt <= size_thr:
                    warning[labels == i] = np.max(warning, axis=(0, 1))
            return warning

        def set_new_lvl(warning, size_thr):
            # correct the max_lvl regions generated before down,
            # until the new regions with it are large enough
            for i in range(np.max(warning, axis=(0, 1)), np.min(warning, axis=(0, 1)), -1):
                level = copy.deepcopy(warning)
                level[warning != i] = 0
                labels = label(warning)
                for j in range(len(np.unique(labels)) + 1):
                    cnt = np.count_nonzero(labels == j)
                    if cnt <= size_thr:
                        warning[labels == j] = i - 1

        warning = warning + 1  # 0 is regarded as background in labelling, + 1 prevents this
        increase_levels(warning, size_thr)
        set_new_lvl(warning, size_thr)
        warning = warning - 1
        return warning
