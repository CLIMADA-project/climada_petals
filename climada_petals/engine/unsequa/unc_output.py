"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define the uncertainty output class for network cascade calculations
"""

__all__ = [
    "UncCascadeOutput",
]

import logging

import numpy as np

from climada.engine.unsequa.unc_output import UncOutput

LOGGER = logging.getLogger(__name__)


class UncCascadeOutput(UncOutput):
    """Uncertainty and sensitivity output of CalcCascade

    Returned by ``CalcCascade.uncertainty``. ``imp_met_unc_df`` holds the
    network statistics of
    :py:func:`climada_petals.engine.networks.nw_utils.make_network_stat` for
    each sample (one row per sample, one column per statistic). The statistics
    have different units, given by ``imp_met_units``:

    - population (``people``): ``total_population``, ``directly_affected``,
      ``indirectly_affected``, ``total_affected``, and for each ci_type with an
      end-user dependency ``{ci_type}_access_loss``,
      ``{ci_type}_no_base_access``, ``{ci_type}_access_undisrupted``;
    - infrastructure, ``{ci_type}_infra_impact`` and ``{ci_type}_infra_total``:
      metres (``m``, length of the edges) for line types (ci_types of the
      edges, e.g. roads), number of nodes (``nodes``) for point types (e.g.
      hospitals).

    ``plot_uncertainty`` labels each statistic with its unit. Statistics that
    do not change over the samples (e.g. ``{ci_type}_infra_total`` when the
    network is fixed) have no meaningful sensitivity indices. Save the output
    with ``to_hdf5`` and load it with ``UncCascadeOutput.from_hdf5``.

    Attributes
    ----------
    imp_met_unc_df : pandas.DataFrame
        Network statistics for each sample.
    imp_met_unit_df : pandas.DataFrame or None
        One row with the unit of each statistic (same columns as
        ``imp_met_unc_df``). None for outputs saved before the units were
        stored: every statistic then has the unit ``unit``.
    imp_met_sens_df : pandas.DataFrame or None
        Sensitivity indices, set by ``CalcCascade.sensitivity``.
    """

    def __init__(
        self,
        samples_df,
        unit,
        imp_met_unc_df,
        imp_met_unit_df=None,
    ):
        """Constructor

        Parameters
        ----------
        samples_df : pandas.DataFrame
            input parameters samples
        unit : str
            unit of the population statistics (``"people"``), also used for
            the statistics without an entry in ``imp_met_unit_df``
        imp_met_unc_df : pandas.DataFrame
            Network statistics for each sample (rows), e.g. the number of
            people losing access to each ci_type (columns), as computed by
            :py:func:`climada_petals.engine.networks.nw_utils.make_network_stat`.
        imp_met_unit_df : pandas.DataFrame, optional
            One row with the unit of each statistic of ``imp_met_unc_df``.
            The default is None (unit ``unit`` for every statistic).
        """
        super().__init__(samples_df, unit)
        self.imp_met_unc_df = imp_met_unc_df
        self.imp_met_unit_df = imp_met_unit_df
        self.imp_met_sens_df = None

    @property
    def imp_met_units(self):
        """Unit of each statistic, as ``{column: unit}``

        Statistics without an entry in ``imp_met_unit_df`` (or all of them,
        if there is no units table) have the unit ``unit``.
        """
        units = {column: self.unit for column in self.imp_met_unc_df.columns}
        if self.imp_met_unit_df is not None and not self.imp_met_unit_df.empty:
            units.update(self.imp_met_unit_df.iloc[0].to_dict())
        return units

    def plot_uncertainty(self, *args, **kwargs):
        """Plot the distribution of each statistic, labelled with its unit

        Same arguments and return value as ``UncOutput.plot_uncertainty``,
        which labels every x-axis with the single unit ``unit``. Here the unit
        of each label is replaced by the unit of its statistic (see
        ``imp_met_units``), e.g. ``road_infra_impact [k m]``.
        """
        axes = super().plot_uncertainty(*args, **kwargs)
        units = self.imp_met_units
        for ax in np.ravel(axes):
            label = ax.get_xlabel()
            # label of UncOutput.plot_uncertainty: "{column} [{prefix} {unit}] "
            unit = units.get(label.split(" [")[0])
            if unit is not None and unit != self.unit:
                ax.set_xlabel(label.replace(f" {self.unit}]", f" {unit}]"))
        return axes

    @staticmethod
    def from_hdf5(filename):
        """Load an output of CalcCascade saved with ``to_hdf5``

        ``UncOutput.from_hdf5`` loads the same data as a base ``UncOutput``,
        without the units of the statistics in the plots.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path of the hdf5 file.

        Returns
        -------
        UncCascadeOutput or None
            The loaded output; None if the file does not exist (as
            ``UncOutput.from_hdf5``). Files saved before the units were
            stored have no units table: every statistic then has the unit
            ``unit`` (logged).

        Raises
        ------
        ValueError
            If the file has no ``imp_met_unc_df`` (not an output of
            CalcCascade).
        """
        unc_output = UncOutput.from_hdf5(filename)
        if unc_output is None:
            return None
        if getattr(unc_output, "imp_met_unc_df", None) is None:
            raise ValueError(
                f"{filename} is not an output of CalcCascade: it has no "
                "imp_met_unc_df."
            )
        cascade_output = UncCascadeOutput.__new__(UncCascadeOutput)
        # copy all loaded attributes (data, unit, sampling and sensitivity
        # settings), so that attributes added by climada are kept too
        cascade_output.__dict__.update(unc_output.__dict__)
        cascade_output.__dict__.setdefault("imp_met_sens_df", None)
        if cascade_output.__dict__.setdefault("imp_met_unit_df", None) is None:
            LOGGER.info(
                "%s has no units table: every statistic has the unit '%s'.",
                filename,
                cascade_output.unit,
            )
        return cascade_output
