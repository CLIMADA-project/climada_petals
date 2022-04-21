import logging
import copy
from dataclasses import dataclass

import numpy as np
import skimage

from climada.util.plot import geo_scatter_categorical

LOGGER = logging.getLogger(__name__)


def dilation(bin_map, size):
    """Dilate binary input map. Enlarges and connects regions of interests. Larger sizes - more area of interest.

    Parameters
    ----------
    bin_map : np.ndarray
        Rectangle 2d map of values which are used to generate the warning.
    size : int
        Size of kernel.

    Returns
    ----------
    np.ndarray
        Generated binary map with enlarged regions of interest.
    """
    return skimage.morphology.dilation(bin_map, skimage.morphology.disk(size))


def erosion(bin_map, size):
    """Erode binary input map. Reduces region of interest and heterogeneity in map. Larger sizes - more reduction.

    Parameters
    ----------
    bin_map : np.ndarray
        Rectangle 2d map of values which are used to generate the warning.
    size : int
        Size of kernel.

    Returns
    ----------
    np.ndarray
        Generated binary map with reduced regions of interest.
    """
    return skimage.morphology.erosion(bin_map, skimage.morphology.disk(size))


def median_filtering(bin_map, size):
    """Smooth out binary map. Larger sizes - smoother regions and more reduction of heterogeneity.

    Parameters
    ----------
    bin_map : np.ndarray
        Rectangle 2d map of values which are used to generate the warning.
    size : int
        Size of kernel.

    Returns
    ----------
    np.ndarray
        Generated binary map with smoothed regions of interest.
    """
    return skimage.filters.median(bin_map, np.ones((size, size)))


ALLOWED_OPERATIONS = {
    'DILATION': dilation,
    'EROSION': erosion,
    'MEDIANFILTERING': median_filtering
}


@dataclass
class FilterData:
    """FilterData data class definition. It stores the relevant information needed during the warning generation.
    The operations and its sizes, as well as the algorithms properties (gradual decrease of warning levels and
    changing of small warning regions formed) are saved.

    Attributes
    ----------
    warn_levels : list
        Warn levels that define the bins in which the input_map will be classified in.
    operations : list
        Operations to be applied in filtering algorithm.
        Possible values: 'DILATION', 'EROSION', 'MEDIANFILTERING'.
    sizes : list
        Size of kernel of every operation given.
    gradual_decr : bool
        Defines whether the highest warn levels should be gradually decreased by its neighboring regions (if True)
        to the lowest level (e.g., level 3, 2, 1, 0)
        or larger steps are allowed (e.g., from warn level 5 directly to 1).
    change_sm : bool
        If True, the levels of too small regions are changed to its surrounding levels.
    size_sm : int
        Defining what too small regions are. Number of coordinates.
    """
    # Explanation of defaults (also maybe transfer to config, talk to Emanuel)
    OPERATIONS = ['DILATION', 'EROSION', 'DILATION', 'MEDIANFILTERING']
    SIZES = [2, 3, 7, 15]
    GRADUAL_DECREASE = True
    CHANGE_SMALL_REGIONS = True
    SIZE_TOO_SMALL = 250

    def __init__(self, warn_levels, operations, sizes, gradual_decr, change_sm, size_sm):
        """Initialize FilterData."""
        self.warn_levels = warn_levels
        self.gradual_decr = gradual_decr
        self.allowed_operations = ALLOWED_OPERATIONS

        if len(operations) != len(sizes):
            LOGGER.warning('For every operation a filter size is needed and the other way around. '
                           'Please input the same number of operations and filter sizes.')
        if not all(item in self.allowed_operations.keys() for item in operations):
            raise ValueError("An input operation is not defined. "
                             "Please select one of %s", self.allowed_operations.keys())
        self.operations = operations
        self.sizes = sizes
        self.change_sm = change_sm
        self.size_sm = size_sm

    @classmethod
    def wind_mch_default(cls, warn_levels, operations=OPERATIONS, sizes=SIZES, gradual_decr=GRADUAL_DECREASE,
                         change_sm=CHANGE_SMALL_REGIONS, size_sm=SIZE_TOO_SMALL):
        """Generate FilterData dataclass with defaults of MCH.

        Returns
        ----------
        warn : Warn
            Generated Warn object including warning, coordinates, warn levels, and metadata.
        """
        return cls(warn_levels, operations, sizes, gradual_decr, change_sm, size_sm)


class Warn:
    """Warn definition. Generate a warning, i.e., 2D map of coordinates with assigned warn levels. Operations,
    their order, and their influence (kernel sizes) can be selected to generate the warning. Further properties can
    be chosen which define the warning generation. The functionality of reducing heterogeneity in a map can be
    applied to different inputs, e.g. MeteoSwiss windstorm data (COSMO data), TCs, impacts, etc.

    Attributes
    ----------
    warning : np.ndarray
        Warning generated by warning generation algorithm. Warn level for every coordinate of input map.
    coord : np.ndarray
        Coordinates of warning map.
    warn_levels : list
        Warn levels that define the bins in which the input_map will be classified in.
        E.g., for windspeeds: [0, 10, 40, 80, 150, 200.0]
    metadata_generation_storm : dict
        Storing metadata on how the warning has been generated and which properties it has.
    """

    def __init__(self, warning, coord, filter_data):
        """Initialize Warn.

        Parameters
        ----------
        warning : np.ndarray
            Warn level for every coordinate of input map.
        coord : np.ndarray
            Coordinates of warning map.
        filter_data : dataclass
            Dataclass consisting information on how to generate the warning (operations and details).
        """
        self.warning = warning
        self.coord = coord
        self.warn_levels = filter_data.warn_levels
        self.metadata_generation_storm = {
            'gradual decrease': filter_data.gradual_decr,
            'change small regions': filter_data.change_sm
        }

    @classmethod
    def from_map(cls, input_map, coord, filter_data):
        """Generate Warn object from map (value (e.g., windspeeds at coordinates).

        Parameters
        ----------
        input_map : np.ndarray
            Rectangle 2d map of values which are used to generate the warning.
        coord : np.ndarray
            Coordinates of warning map.
        filter_data : dataclass
            Dataclass consisting information on how to generate the warning (operations and details).

        Returns
        ----------
        warn : Warn
            Generated Warn object including warning, coordinates, warn levels, and metadata.
        """
        binned_map = cls.bin_map(input_map, filter_data.warn_levels)
        warning = cls.generate_warn_map(binned_map, filter_data)
        if filter_data.change_sm:
            warning = cls.change_small_regions(warning, filter_data.size_sm)
        return cls(warning, coord, filter_data)

    @staticmethod
    def bin_map(input_map, levels):
        """Bin every value of input map into given levels.

        Parameters
        ----------
        input_map : np.ndarray
            Array containing data to generate binned map of.
        levels : list
            List with levels to bin input map.

        Returns
        ----------
        binned_map : np.ndarray
            Map of binned values in levels, same shape as input map.
        """
        if np.min(input_map) < np.min(levels):
            LOGGER.warning('Values of input map are smaller than defined levels. '
                           'Please set the levels lower or check input map.')
        if np.max(input_map) > np.max(levels):
            LOGGER.warning('Values of input map are larger than defined levels. '
                           'Please set the levels higher or check input map.')
        return np.digitize(input_map, levels) - 1  # digitize lowest bin is 1

    @staticmethod
    def filtering(binary_map, filter_data):
        """For the current warn level, apply defined operations in filter data on the input binary map.

        Parameters
        ----------
        binary_map : np.ndarray
            Binary 2D array, where 1 corresponds to current (and higher if grad_decrease) warn level and 0 else.
        filter_data : dataclass
            Dataclass consisting information on how to generate the warning (operations and details).

        Returns
        ----------
        binary_curr_lvl : np.ndarray
            Warning map consisting formed warning regions of current warn level, same shape as input map.
        """
        for fl_size, op in zip(filter_data.sizes, filter_data.operations):
            binary_map = filter_data.allowed_operations[op](binary_map, fl_size)

        return binary_map

    @staticmethod
    def generate_warn_map(bin_map, filter_data):
        """Generate warning map of binned map. The filter algorithm reduces heterogeneity in the map (erosion) and
        makes sure warn regions of higher warn levels warn regions large enough (dilation). With the median
        filtering the generated warning is smoothed out without blurring.

        Parameters
        ----------
        bin_map : np.ndarray
            Map of binned values in warn levels. Hereof a warning with contiguous regions is formed.
        filter_data : dataclass
            Dataclass consisting information on how to generate the warning (operations and details).

        Returns
        ----------
        warn_regions : np.ndarray
            Warning map consisting formed warning regions, same shape as input map.
        """
        unq = np.unique(bin_map)
        if len(unq) == 1:
            return np.zeros_like(bin_map) + unq[0]
        max_warn_level = np.max(bin_map)
        min_warn_level = np.min(bin_map)

        warn_map = np.zeros_like(bin_map) + min_warn_level
        for curr_lvl in range(max_warn_level, min_warn_level, -1):
            if filter_data.gradual_decr:
                pts_curr_lvl = np.bitwise_or(warn_map > curr_lvl, bin_map >= curr_lvl)
            else:
                pts_curr_lvl = bin_map == curr_lvl
            binary_curr_lvl = np.where(pts_curr_lvl, curr_lvl, 0)  # set bool np.ndarray to curr_lvl (if True) or 0

            warn_reg = Warn.filtering(binary_curr_lvl, filter_data)
            warn_map = np.maximum(warn_map, warn_reg)  # keep warn regions of higher levels by taking maximum

        return warn_map

    @staticmethod
    def increase_levels(warn, size):
        """Increase warn levels of too small regions to max warn level of this warning.

        Parameters
        ----------
        warn : np.ndarray
            Warning map of which too small regions are changed to surrounding. Warn levels are all +1.
        size : int
            Threshold defining too small regions (number of coordinates).

        Returns
        ----------
        warn : np.ndarray
            Warning map where too small regions are of the higher level occurring. Warn levels are all +1.
        """
        labels = skimage.measure.label(warn)
        for l in np.unique(labels):
            cnt = np.count_nonzero(labels == l)
            if cnt <= size:
                warn[labels == l] = np.max(warn, axis=(0, 1))
        return warn

    @staticmethod
    def reset_levels(warn, size):
        """Set warn levels of too small regions to highest surrounding warn level. Therefore, decrease warn levels of
        too small regions, until no too small regions can be detected.

        Parameters
        ----------
        warn : np.ndarray
            Warning map of which too small regions are changed to surrounding. Warn levels are all +1.
        size : int
            Threshold defining too small regions (number of coordinates).

        Returns
        ----------
        warn : np.ndarray
            Warning map where too small regions are changed to neighborhood. Warn levels are all +1.
        """
        for i in range(np.max(warn), np.min(warn), -1):
            level = copy.deepcopy(warn)
            level[warn != i] = 0
            labels = skimage.measure.label(warn)
            for l in np.unique(labels):
                cnt = np.count_nonzero(labels == l)
                if cnt <= size:
                    warn[labels == l] = i - 1
        return warn

    @staticmethod
    def change_small_regions(warning, size):
        """Change formed warning regions smaller than defined threshold from current warn level to surrounding warn
        level.

        Parameters
        ----------
        warning : np.ndarray
            Warning map of which too small regions are changed to surrounding.
        size : int
            Threshold defining too small regions (number of coordinates).

        Returns
        ----------
        warning : np.ndarray
            Warning without too small regions, same shape as input map.
        """
        warning = warning + 1  # 0 is regarded as background in labelling, + 1 prevents this
        warning = Warn.increase_levels(warning, size)
        warning = Warn.reset_levels(warning, size)
        warning = warning - 1
        return warning

    def plot_warning(self, var_name='Warn Levels', title='Categorical Warning Map', cat_name=None, adapt_fontsize=True, **kwargs):
        return geo_scatter_categorical(self.warning.flatten(), self.coord, var_name, title, cat_name, adapt_fontsize,
                                       **kwargs)
