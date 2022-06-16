#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:35:55 2022

@author: ckropf
"""

import pandas as pd
from climada import CONFIG
from pathlib import Path

isc_gem_cat_file='/Users/ckropf/Documents/Climada/climada_petals/climada_petals/hazard/test/data/isc-gem-cat.csv'
isc_gem_cat = pd.read_csv(isc_gem_cat_file, delimiter=' , ', comment='#', header=None, engine='python')

with open(isc_gem_cat_file) as fl:
    for ln in fl.readlines():
        if ln[0] != '#': break
        pr = ln
isc_gem_cat.columns = [x.strip() for x in pr[1:].split(',')]

from climada.hazard import Centroids
centroids_file='/Users/ckropf/Documents/Climada/climada_petals/climada_petals/hazard/test/data/NZL_NewZealand_centroids.mat'
centroids=Centroids.from_mat(centroids_file)
centroids.set_meta_to_lat_lon()

from climada_petals.hazard import Earthquake


quake = Earthquake.from_Mw_depth(isc_gem_cat, centroids)