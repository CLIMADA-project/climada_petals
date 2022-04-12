import logging
import copy
from dataclasses import dataclass

import numpy as np
import skimage

LOGGER = logging.getLogger(__name__)


@dataclass
class FilterData:
    """FilterData data class.

    The FilterData data class stores the relevant information needed during filtering.

    Attributes
    ----------
    see @classmethod's attributes
    """

    def __init__(self, thresholds, expand, operations, sizes):
        """Initialize FilterData."""
        self.thresholds = thresholds
        self.expand = expand
        if len(operations) != len(sizes):
            LOGGER.warning('For every operation a filter size is needed and the other way around. '
                           'Please input the same number of operations and filter sizes.')
        self.operations = operations
        self.sizes = sizes


class Warn:
    """Warn class.

    The Warn class generates from given data contiguous areas and reduces heterogeneity.

    Attributes
    ----------
    filter_data : dataclass
        dataclass consisting of thresholds, whether the regions should be expanded or not,
        the operations and their sizes to be applied during filtering
    nr_thresholds : int
        Number of thresholds defined
    warning : np.array
        Warn level for every point
    """
    # Explanation of defaults (also maybe transfer to config, talk to Emanuel)
    OPERATIONS = ['DILATION', 'EROSION', 'DILATION', 'MEDIANFILTERING']
    SIZES = [2, 3, 7, 15]
    EXPAND = True

    def __init__(self, warning, filter_data, coord):
        """Initialize Warn."""
        self.warning = warning
        self.filter_data = filter_data
        self.coord = coord
        self.nr_thresholds = len(self.filter_data.thresholds) - 1

    @classmethod
    def from_map(cls, data, thresholds, coord, expand=EXPAND, operations=OPERATIONS, sizes=SIZES):
        """Generate Warn object from np.array.

        Parameters
        ----------
        data : np.array
            2d np.array containing data to generate warning of.
        thresholds : np.array
            Thresholds where data is binned.
        expand : Bool
            If true, regions of higher warning levels expand gradually into lower levels.
        operations : np.array
            Type of operations to be applied in filtering algorithm.
        sizes : np.array
            Size of kernel of every operation given.

        Returns
        ----------
        warn : Warn
            Generated Warn object including warning
        """
        filter_data = FilterData(thresholds, expand, operations, sizes)
        data_thrs = cls.threshold_data(data, filter_data)
        warning = cls.filter_algorithm(data_thrs, filter_data)
        warn = cls(warning, filter_data, coord)
        return warn

    @staticmethod
    def threshold_data(data, filter_data):
        """Threshold data into given thresholds.

        Parameters
        ----------
        data : np.array
            Array containing data to generate warning of.

        Returns
        ----------
        m_thrs : np.array
            Array of thresholded data
        """
        if np.max(data) > np.max(filter_data.thresholds) or np.min(data) < np.min(filter_data.thresholds):
            LOGGER.warning('Values of data array are smaller/larger than defined thresholds. '
                           'Please redefine thresholds.')
        m_thrs = np.digitize(data, filter_data.thresholds) - 1  # digitize lowest bin is 1
        return m_thrs

    @staticmethod
    def filtering(d, warn_reg, curr_lvl, filter_data):
            if filter_data.expand:  # select points where level is >= level under observation -> expands regions
                pts_curr_lvl = np.bitwise_or(warn_reg, d >= curr_lvl)
            else:  # select points where level is == level under observation -> not expanding regions
                pts_curr_lvl = d == curr_lvl
            reg_curr_lvl = np.where(pts_curr_lvl, curr_lvl, 0)  # set everything but pts of current level to 0

            for i in range(len(filter_data.operations)):
                if filter_data.operations[i] == 'DILATION':
                    reg_curr_lvl = skimage.morphology.dilation(reg_curr_lvl,
                                                               skimage.morphology.disk(filter_data.sizes[i]))
                elif filter_data.operations[i] == 'EROSION':
                    reg_curr_lvl = skimage.morphology.erosion(reg_curr_lvl,
                                                              skimage.morphology.disk(filter_data.sizes[i]))
                elif filter_data.operations[i] == 'MEDIANFILTERING':
                    filter_med = np.ones((filter_data.sizes[i], filter_data.sizes[i]))
                    reg_curr_lvl = skimage.filters.median(reg_curr_lvl, filter_med)
                else:
                    LOGGER.warning("The operation is not defined. "
                                   "Please select 'EROSION', 'DILATION', or 'MEDIANFILTERING'.")
            return reg_curr_lvl

    @staticmethod
    def filter_algorithm(d_thrs, filter_data):
        """Generate contiguous regions of thresholded data.

        Parameters
        ----------
        d_thrs : np.array
            Thresholded data to generate contiguous regions of.
        """


        max_warn_level = np.max(d_thrs)
        if max_warn_level == 0:
            return np.zeros(d_thrs.shape)

        warn_regions = 0
        # iterate from highest level to lowest (0 not necessary, because rest is level 0)
        for j in range(max_warn_level, 0, -1):
            w_l = Warn.filtering(d_thrs, warn_regions, j, filter_data)
            # keep warn regions of higher levels by taking maximum
            warn_regions = np.maximum(warn_regions, w_l)
        return warn_regions

    @staticmethod
    def increase_levels(warn, size):
            # increase levels of too small regions to max level of this warning
            labels = skimage.measure.label(warn)
            for i in range(np.max(labels)):
                cnt = np.count_nonzero(labels == i)
                if cnt <= size:
                    warn[labels == i] = np.max(warn, axis=(0, 1))
            return warn

    @staticmethod
    def set_new_lvl(warn, size):
            # correct the max_lvl regions generated before down,
            # until the new regions with it are large enough
            for i in range(np.max(warn, axis=(0, 1)), np.min(warn, axis=(0, 1)), -1):
                level = copy.deepcopy(warn)
                level[warn != i] = 0
                labels = skimage.measure.label(warn)
                for j in range(len(np.unique(labels)) + 1):
                    cnt = np.count_nonzero(labels == j)
                    if cnt <= size:
                        warn[labels == j] = i - 1
            return warn

    @staticmethod
    def remove_small_regions(warning, size_thr):
        """Remove regions smaller than defined threshold from warning.

        Parameters
        ----------
        warning : np.array
            Warning where regions should be removed from.
        size_thr : int
            Threshold defining too small regions (number of grid points).

        Returns
        ----------
        warning : np.array
            Warning without too small regions.
        """
        warning = warning + 1  # 0 is regarded as background in labelling, + 1 prevents this
        warning = Warn.increase_levels(warning, size_thr)
        warning = Warn.set_new_lvl(warning, size_thr)
        warning = warning - 1
        return warning
