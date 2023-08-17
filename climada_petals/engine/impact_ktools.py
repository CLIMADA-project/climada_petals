"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define Impact and ImpactFreqCurve classes.
"""

#__all__ = ['ImpactFreqCurve', 'ImpactKtools']

import logging
import numpy as np
import pandas as pd

#imports by Robert Blass
from pathlib import Path
import platform
import time
import subprocess

from climada.engine.impact import Impact
from climada.entity.exposures.base import INDICATOR_CENTR

LOGGER = logging.getLogger(__name__)

class ImpactKtools(Impact):
    """Impact definition. Compute from an entity (exposures and impact
    functions) and hazard.

    Attributes:
        tag (dict): dictionary of tags of exposures, impact functions set and
            hazard: {'exp': Tag(), 'if_set': Tag(), 'haz': TagHazard()}
        event_id (np.array): id (>0) of each hazard event
        event_name (list): name of each hazard event
        date (np.array): date of events
        coord_exp (np.ndarray): exposures coordinates [lat, lon] (in degrees)
        eai_exp (np.array): expected annual impact for each exposure
        at_event (np.array): impact for each hazard event
        frequency (np.arrray): annual frequency of event
        tot_value (float): total exposure value affected
        aai_agg (float): average annual impact (aggregated)
        unit (str): value unit used (given by exposures unit)
        imp_mat (sparse.csr_matrix): matrix num_events x num_exp with impacts.
            only filled if save_mat is True in calc()
        sd_at_event (np.ndarray): average standard deviation per event
        sd_eai_exp (np.ndarray): average standard deviation per area
    """

    def __init__(self):
        """Empty initialization."""
        super().__init__()
        self.sd_at_event = np.array([])
        self.sd_eai_exp = np.array([])

    def data_conversion_ktools(self, indicator, data_path, path_exe_ktools, max_information=0):
        """Generate binary file from the corresponding csv file.

        Parameters
            indicator (String): indicates given information and file to generate
            data_path (Path): path where the files are saved
            path_exe_ktools (Path): path to ktools executables
            max_information (Integer, optional): maximum used by ktools in file header
                for footprint: maximum intensity bin index
                for vulnerability: maximum damage bin index
        """
        if(platform.system() == 'Windows'):
            if indicator == 'events':
                cmd = (path_exe_ktools + 'evetobin < events.csv > events.bin')
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=data_path)
                print(cmd)
            elif indicator == 'damage_bin':
                cmd = (path_exe_ktools + 'damagebintobin < damage_bin_dict.csv > damage_bin_dict.bin')
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=data_path)
                print(cmd)
            elif indicator == 'items':
                cmd = (path_exe_ktools + 'itemtobin < items.csv > items.bin')
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=data_path)
                print(cmd)
            elif indicator == 'coverages':
                cmd = (path_exe_ktools + 'coveragetobin < coverages.csv > coverages.bin')
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=data_path)
                print(cmd)
            elif indicator == 'footprint':
                cmd = (path_exe_ktools + 'footprinttobin -i' + str(max_information) + ' < footprint.csv')
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=data_path)
                print(cmd)
            elif indicator == 'vulnerability':
                cmd = (path_exe_ktools + 'vulnerabilitytobin -d' + str(max_information) + ' < vulnerability.csv > vulnerability.bin')
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=data_path)
                print(cmd)
            elif indicator == 'occurrences':
                cmd = (path_exe_ktools + 'occurrencetobin -P' + str(max_information) + ' -D  < occurrence.csv > occurrence.bin')
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=data_path)
                print(cmd)
            elif indicator == 'return_period':
                cmd = (path_exe_ktools + 'returnperiodtobin < returnperiods.csv > returnperiods.bin')
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=data_path)
                print(cmd)
            elif indicator == 'periods':
                cmd = (path_exe_ktools + 'periodstobin < periods.csv > periods.bin')
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=data_path)
                print(cmd)
            elif indicator == 'gul_summary':
                cmd = (path_exe_ktools + 'gulsummaryxreftobin < gulsummaryxref.csv > gulsummaryxref.bin')
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=data_path)
                print(cmd)
        else:
            if indicator == 'events':
                cmd = 'evetobin < events.csv > events.bin'
                subprocess.Popen(cmd, shell=True, cwd=data_path)
                print(cmd)
            elif indicator == 'damage_bin':
                cmd = 'damagebintobin < damage_bin_dict.csv > damage_bin_dict.bin'
                subprocess.Popen(cmd, shell=True, cwd=data_path)
                print(cmd)
            elif indicator == 'items':
                cmd = 'itemtobin < items.csv > items.bin'
                subprocess.Popen(cmd, shell=True, cwd=data_path)
                print(cmd)
            elif indicator == 'coverages':
                cmd = 'coveragetobin < coverages.csv > coverages.bin'
                subprocess.Popen(cmd, shell=True, cwd=data_path)
                print(cmd)
            elif indicator == 'footprint':
                cmd = 'footprinttobin -i' + str(max_information) + ' < footprint.csv'
                subprocess.Popen(cmd, shell=True, cwd=data_path)
                print(cmd)
            elif indicator == 'vulnerability':
                cmd = 'vulnerabilitytobin -d ' + str(max_information) + ' < vulnerability.csv > vulnerability.bin'
                subprocess.Popen(cmd, shell=True, cwd=data_path)
                print(cmd)
            elif indicator == 'occurences':
                cmd = 'occurrencetobin -P' + str(max_information) + ' -D < occurrence.csv > occurrence.bin'
                subprocess.Popen(cmd, shell=True, cwd=data_path)
                print(cmd)
            elif indicator == 'return_period':
                cmd = 'returnperiodtobin < returnperiods.csv > returnperiods.bin'
                subprocess.Popen(cmd, shell=True, cwd=data_path)
                print(cmd)
            elif indicator == 'periods':
                cmd = 'periodstobin < periods.csv > periods.bin'
                subprocess.Popen(cmd, shell=True, cwd=data_path)
                print(cmd)
            elif indicator == 'gul_summary':
                cmd = 'gulsummaryxreftobin < gulsummaryxref.csv > gulsummaryxref.bin'
                subprocess.Popen(cmd, shell=True, cwd=data_path)
                print(cmd)


    def generate_events(self, hazard, data_path, path_exe_ktools):
        df = pd.DataFrame(data = {'event_id': hazard.event_id}, dtype='int32')
        df.to_csv(data_path / 'events.csv', index=False)
        self.data_conversion_ktools('events', data_path, path_exe_ktools)

    def generate_damage_bins(self, imp_func, nDamageBins, data_path, path_exe_ktools):
        """Divides intensities and corresponding mdr-values into damage bins.
           Save as csv  and binary file.

        Parameters
            haz_imp (ImpactFuncSet): impact functions
            nDamageBins (Integer): number of bins
            data_path (Path): path where the files are saved
            path_exe_ktools (Path): path to ktools executables

        Returns
            discrete_intensity_range (np.array): discretized intensity range
        """

        discrete_intensity_range = np.linspace(min(imp_func.intensity),
                                               max(imp_func.intensity), nDamageBins)
        #compute damage bins
        discrete_mdr_range = imp_func.calc_mdr(discrete_intensity_range)
        damage_bin_from = discrete_mdr_range
        damage_bin_to = np.append(discrete_mdr_range[1:],
                                  imp_func.calc_mdr(max(imp_func.intensity)))
        damage_bin_interpolation = 0.5 * (damage_bin_from + damage_bin_to)
        #damage_bin_from = damage_bins[:-1]
        #damage_bin_to = damage_bins[1:]
        #damage_bin_interpolation = 0.5 * (damage_bin_from + damage_bin_to)
        #nDamageBins = len(damage_bin_from)

        df = pd.DataFrame(data = {'bin_index': range(1, nDamageBins + 1),
                                  'bin_from': damage_bin_from,
                                  'bin_to': damage_bin_to,
                                  'interpolation': damage_bin_interpolation,
                                  #interval_type irrelevant
                                  'interval_type': np.full(shape=nDamageBins,
                                                           fill_value=1)},
                          dtype='int32')
        df.to_csv(data_path / 'damage_bin_dict.csv', index=False)
        self.data_conversion_ktools('damage_bin', data_path, path_exe_ktools)

    def generate_footprint(self, hazard, n_intensity_bins, prob, data_path, path_exe_ktools):
        """Discretize hazard intensities into bins given by discrete_intensity_range.
           Save as csv and binary file.

        Parameters
            hazard (Hazard): hazard
            discrete_intensity_range (np.array): discretized intensity range
            data_path (Path): path where the files are saved
            path_exe_ktools (Path): path to ktools executables

        Returns
            discrete_hazard_intensity (np.array): discretized hazard intensity
        """
        hazard_event_id, centroid_idx, hazard_intensity_values, footprint_prob = [], [], [], []
        p = 0
        #need probability for every intensity bin, event and centroid
        for r in range(hazard.intensity.shape[0]):
            for ind in range(hazard.intensity.indptr[r], hazard.intensity.indptr[r+1]):
                for k in range(1, n_intensity_bins+1):
                    hazard_event_id.append(hazard.event_id[r])
                    #centroid index in ktools starts at 1 -> +1. Also accounted for in generate_items
                    centroid_idx.append(hazard.intensity.indices[ind] + 1)
                    hazard_intensity_values.append(k)
                    footprint_prob.append(prob[p])
                    p += 1

        df = pd.DataFrame(data = {'event_id': hazard_event_id,
                                  'areaperil_id': centroid_idx,
                                  'intensity_bin_index': hazard_intensity_values,
                                  'prob': footprint_prob},
                          dtype='int32')
        df.to_csv(data_path / 'footprint.csv', index=False)
        self.data_conversion_ktools('footprint', data_path, path_exe_ktools,
                                    max(hazard_intensity_values))

    def generate_vulnerability(self, imp_ids, n_damage_bins, n_intensity_bins, vulnerability_ids, prob, data_path, path_exe_ktools):
        vul_id, intens_idx, dam_idx, vul_prob = [], [], [], []
        p = 0
        #probability value needs to be defined for every damage and intensity bin combination
        for i in vulnerability_ids:
            for j in range(1, n_intensity_bins+1):
                for k in range(1, n_damage_bins+1):
                    vul_id.append(i)
                    intens_idx.append(j)
                    dam_idx.append(k)
                    vul_prob.append(prob[p])
                    p += 1

        df = pd.DataFrame(data = {'vulnerability_id': vul_id,
                                  'intensity_bin_index': intens_idx,
                                  'damage_bin_index': dam_idx,
                                  'prob': vul_prob},
                          dtype='int32')
        df.to_csv(data_path / 'vulnerability.csv', index=False)
        self.data_conversion_ktools('vulnerability', data_path, path_exe_ktools, max(dam_idx))

    def generate_items(self, exposures, vul_id_per_area, data_path, path_exe_ktools):
        l = exposures.centr_TC.size
        coverage_id = range(1, l+1)
        itm = pd.DataFrame(data = {'item_id': range(1, l+1),
                                   'coverage_id': coverage_id,
                                   #centroid index in ktools starts at 1. Also accounted for in generate footprint
                                   'areaperil_id': (exposures.centr_TC + 1),
                                   'vulnerability_id': vul_id_per_area,
                                   'group_id': range(1, l+1)}, #no influence
                           dtype='int32')
        itm.to_csv(data_path / 'items.csv', index=False)
        self.data_conversion_ktools('items', data_path, path_exe_ktools)

    def exp_impact_ktools(self, exposures, hazard, data_path):
        """Compute impact for input exposures and hazard.

        Parameters
            hazard (Hazard): hazard instance
            exposures (Exposures): exposures instance
            data_path (Path): path to output file
        """
        #import output file of ktools (here eltcalc). Adapt if other output file generated
        time.sleep(5)
        if(platform.system() == 'Windows'):
            text = str(data_path) + "\eltcalc.csv"
            elt_file = pd.read_csv(text, index_col=None)
        else:
            text = str(data_path) + "/eltcalc.csv"
            elt_file = pd.read_csv(text, index_col=None)

        mean_imp = elt_file['mean'].to_numpy()
        sidx = elt_file['type'].to_numpy() #if 2: mean and sd by sampling
        eve_id = elt_file['event_id'].to_numpy()
        cov = elt_file['summary_id'].to_numpy()
        sd = elt_file['standard_deviation'].to_numpy()

        cnt = np.zeros(len(self.at_event))
        ind = 0
        for i in hazard.event_id:
            for j in range(len(mean_imp)):
                if eve_id[j] == i and sidx[j] == 2:
                    self.at_event[ind] += mean_imp[j]
                    self.sd_at_event[ind] += sd[j]
                    cnt[ind] += 1
            ind += 1
        #average sd over areas
        self.sd_at_event = np.divide(self.sd_at_event, cnt, out = self.sd_at_event, where = self.sd_at_event != 0)

        cnt = np.zeros(len(self.eai_exp))
        ind = 0
        for i in range(1, len(exposures.centr_TC)+1):
            for j in range(len(mean_imp)):
                if cov[j] == i and sidx[j] == 2:
                    self.eai_exp[ind] += mean_imp[j]  * hazard.frequency[hazard.event_id == eve_id[j]]
                    self.sd_eai_exp[ind] += sd[j]
                    cnt[ind] += 1
            ind += 1
        #average sd over events
        self.sd_eai_exp = np.divide(self.sd_eai_exp, cnt, out = self.sd_eai_exp, where = self.sd_eai_exp != 0)

    def calc_ktools(self, exposures, impact_funcs, hazard, def_bins, path_exe_ktools='/usr/local/bin/', save_mat=False):
        """Compute impact and its uncertainty of an hazard to exposures with ktools.
        Further information can be found in the Term paper of Robert Blass:
        https://doi.org/10.3929/ethz-b-000480061

        Parameters:
            exposures (Exposures): exposures
            impact_funcs (ImpactFuncSet): impact functionss
            hazard (Hazard): hazard
            nDamageBins (Integer, optional): number of damage bins to generate
            path_exe_ktools (Path, optional): path to ktools executables
            save_mat (bool, optional): self impact matrix: events x exposures
        """
        # 1. Assign centroids to each exposure if not done
        assign_haz = INDICATOR_CENTR + hazard.haz_type
        exposures.assign_centroids(hazard)


        # 2. Initialize values
        self.unit = exposures.value_unit
        self.event_id = hazard.event_id
        self.event_name = hazard.event_name
        self.date = hazard.date
        self.coord_exp = np.stack([exposures.gdf.latitude.values,
                                   exposures.gdf.longitude.values], axis=1)
        self.frequency = hazard.frequency
        self.at_event = np.zeros(hazard.intensity.shape[0])
        self.eai_exp = np.zeros(exposures.gdf.value.size)
        self.sd_at_event = np.zeros(hazard.intensity.shape[0])
        self.sd_eai_exp = np.zeros(exposures.gdf.value.size)
        #self.tag = {'exp': exposures.tag, 'if_set': impact_funcs.tag, 'haz': hazard.tag}
        self.crs = exposures.crs

        # Select exposures with positive value and assigned centroid
        exp_idx = np.where((exposures.gdf.value > 0) & (exposures.gdf[assign_haz] >= 0))[0]
        if exp_idx.size == 0:
            LOGGER.warning("No affected exposures.")

        num_events = hazard.intensity.shape[0]
        LOGGER.info('Calculating damage for %s assets (>0) and %s events.',
                    exp_idx.size, num_events)

        #build folder structure according to ktools
        #Unix as default given
        if(platform.system() == 'Windows'):
            path_exe_ktools = "C:/msys32/mingw64/bin/"
        climada_path = Path.home()
        #exist_ok=True: overwrite dictionary if already existing
        Path.mkdir(climada_path / 'climada', exist_ok=True)
        input_data_path = climada_path / 'climada' / 'ktools_data'
        Path.mkdir(input_data_path, exist_ok=True)
        Path.mkdir(input_data_path / 'input', exist_ok=True)
        ktools_input_dir = input_data_path / 'input'
        Path.mkdir(input_data_path / 'static', exist_ok=True)
        ktools_static_dir = input_data_path / 'static'
        Path.mkdir(input_data_path / 'work', exist_ok=True)
        ktools_work_dir = input_data_path / 'work'
        Path.mkdir(ktools_work_dir / 'sum', exist_ok=True)

        #generate necessary binary files for ktools
        fun_id = exposures.gdf[exposures.get_impf_column(hazard.haz_type)][0]
        [imp_fun] = impact_funcs.get_func(fun_id = fun_id)
        #translate climada, save in csv files and conversion to binary files
        self.generate_events(hazard, ktools_input_dir, path_exe_ktools)
        self.generate_damage_bins(imp_fun, def_bins.get('damage_bins'), ktools_static_dir, path_exe_ktools)
        self.generate_coverages(exposures, ktools_input_dir, path_exe_ktools)
        self.generate_items(exposures, def_bins.get('vul_id_per_area'), ktools_input_dir, path_exe_ktools)
        self.generate_footprint(hazard, def_bins.get('n_intensity_bins'),
                                def_bins.get('footprint_prob'), ktools_static_dir, path_exe_ktools)

        [imp_ids] = np.unique(exposures.if_TC.values)
        self.generate_vulnerability(imp_ids, def_bins.get('n_damage_bins'),
                                    def_bins.get('n_intensity_bins'), def_bins.get('vulnerability_ids'),
                                    def_bins.get('vulnerability_prob'), ktools_static_dir, path_exe_ktools)
        self.generate_occurence(hazard, ktools_input_dir, path_exe_ktools)
        self.generate_gul_summary(exposures, ktools_input_dir, path_exe_ktools)

        #generate output files of ktools
        self.compute_impact_ktools(input_data_path, path_exe_ktools, def_bins.get('sample_size'))
        #calculate output information
        self.exp_impact_ktools(exposures, hazard, input_data_path)
