"""
Impact function calibration functionalities:
    Optimization and manual calibration

based in climada.engine.calib_opt.py and extended by Timo to include
bayesian optimization
"""

import datetime as dt
import copy
import itertools
import logging
import numpy as np
import pandas as pd
import warnings
from scipy import interpolate
from scipy.optimize import minimize
from itertools import combinations
import matplotlib.pyplot as plt

from climada.engine import ImpactCalc, Impact
from climada.entity import ImpactFuncSet, ImpfTropCyclone, ImpactFunc
from climada.engine.impact_data import emdat_impact_yearlysum, emdat_impact_event
try:
    from bayes_opt import BayesianOptimization, UtilityFunction
except:
    print('Could not find bayes_opt. Module Calib_opt will not work.')
LOGGER = logging.getLogger(__name__)



def calib_instance(hazard, exposure, impact_func, df_out=pd.DataFrame(),
                   yearly_impact=False, return_cost='False'):

    """calculate one impact instance for the calibration algorithm and write
        to given DataFrame

        Parameters
        ----------
        hazard : Hazard
        exposure : Exposure
        impact_func : ImpactFunc
        df_out : Dataframe, optional
            Output DataFrame with headers of columns defined and optionally with
            first row (index=0) defined with values. If columns "impact",
            "event_id", or "year" are not included, they are created here.
            For calibration the column "impact_scaled" must be included.
            Data like reported impacts or impact function parameters can be
            given here; values are preserved.
        yearly_impact : boolean, optional
            if set True, impact is returned per year, not per event
        return_cost : str, optional
            if not 'False' but any of 'R2', 'logR2',
            cost is returned instead of df_out

        Returns
        -------
        df_out: DataFrame
            DataFrame with modelled impact written to rows for each year
            or event.
    """
    ifs = ImpactFuncSet([impact_func])
    impacts = ImpactCalc(exposures=exposure, impfset=ifs, hazard=hazard)\
              .impact(assign_centroids=False)

    if yearly_impact:  # impact per year
        iys = impacts.impact_per_year(all_years=True)
        # Loop over whole year range:
        if df_out.empty | df_out.index.shape[0] == 1:
            for cnt_, year in enumerate(np.sort(list((iys.keys())))):
                if cnt_ > 0:
                    df_out.loc[cnt_] = df_out.loc[0]  # copy info from first row
                if year in iys:
                    df_out.loc[cnt_, 'impact_CLIMADA'] = iys[year]
                else:
                    df_out.loc[cnt_, 'impact_CLIMADA'] = 0.0
                df_out.loc[cnt_, 'year'] = year
        else:
            years_in_common = df_out.loc[df_out['year'].isin(np.sort(list((iys.keys())))), 'year']
            for cnt_, year in years_in_common.iteritems():
                df_out.loc[df_out['year'] == year, 'impact_CLIMADA'] = iys[year]


    else:  # impact per event
        if df_out.empty | df_out.index.shape[0] == 1:
            for cnt_, impact in enumerate(impacts.at_event):
                if cnt_ > 0:
                    df_out.loc[cnt_] = df_out.loc[0]  # copy info from first row
                df_out.loc[cnt_, 'impact_CLIMADA'] = impact
                df_out.loc[cnt_, 'event_id'] = int(impacts.event_id[cnt_])
                df_out.loc[cnt_, 'event_name'] = impacts.event_name[cnt_]
                df_out.loc[cnt_, 'year'] = \
                    dt.datetime.fromordinal(impacts.date[cnt_]).year
                df_out.loc[cnt_, 'date'] = impacts.date[cnt_]
        elif df_out.index.shape[0] == impacts.at_event.shape[0]:
            for cnt_, (impact, ind) in enumerate(zip(impacts.at_event, df_out.index)):
                df_out.loc[ind, 'impact_CLIMADA'] = impact
                df_out.loc[ind, 'event_id'] = int(impacts.event_id[cnt_])
                df_out.loc[ind, 'event_name'] = impacts.event_name[cnt_]
                df_out.loc[ind, 'year'] = \
                    dt.datetime.fromordinal(impacts.date[cnt_]).year
                df_out.loc[ind, 'date'] = impacts.date[cnt_]
        else:
            raise ValueError('adding simulated impacts to reported impacts not'
                             ' yet implemented. use yearly_impact=True or run'
                             ' without init_impact_data.')
    if return_cost:
        return calib_cost_calc(df_out, return_cost)
    else:
        return df_out

def init_impf(impf_name_or_instance, param_dict,intensity_range, df_out=pd.DataFrame(index=[0])):
    """create an ImpactFunc based on the parameters in param_dict using the
    method specified in impf_parameterisation_name and document it in df_out.

    Parameters
    ----------
    impf_name_or_instance : str or ImpactFunc
        method of impact function parameterisation e.g. 'emanuel' or an
        instance of ImpactFunc
    param_dict : dict, optional
        dict of parameter_names and values
        e.g. {'v_thresh': 25.7, 'v_half': 70, 'scale': 1}
        or {'mdd_shift': 1.05, 'mdd_scale': 0.8, 'paa_shift': 1, paa_scale': 1}
    intensity_range : array
        tuple of 3 intensity numbers along np.arange(min, max, step)
    Returns
    -------
    imp_fun : ImpactFunc
        The Impact function based on the parameterisation
    df_out : DataFrame
        Output DataFrame with headers of columns defined and with first row
        (index=0) defined with values. The impact function parameters from
        param_dict are represented here.
    """
    impact_func_final = None
    if isinstance(impf_name_or_instance, str):
        if impf_name_or_instance == 'emanuel':
            impact_func_final = ImpfTropCyclone.from_emanuel_usa(**param_dict)
            impact_func_final.haz_type = 'TC'
            impact_func_final.id = 1
            df_out['impact_function'] = impf_name_or_instance
        if impf_name_or_instance == 'emanuel_HL':
            impact_func_final = get_emanuel_impf(
                **param_dict,intensity=intensity_range,haz_type='HL')
            df_out['impact_function'] = impf_name_or_instance
        elif impf_name_or_instance == 'sigmoid_HL':
            assert('L' in param_dict.keys() and 'k' in param_dict.keys() and 'x0' in param_dict.keys()) 
            impact_func_final = ImpactFunc.from_sigmoid_impf(
                **param_dict,intensity=intensity_range)#,haz_type='HL')
            impact_func_final.haz_type = 'HL'
            if intensity_range[0]==0 and not impact_func_final.mdd[0]==0:
                warnings.warn('sigmoid impact function has non-zero impact at intensity 0. Setting impact to 0.')
                impact_func_final.mdd[0]=0
            df_out['impact_function'] = impf_name_or_instance

    elif isinstance(impf_name_or_instance, ImpactFunc):
        impact_func_final = change_impf(impf_name_or_instance, param_dict)
        df_out['impact_function'] = ('given_' +
                                     impact_func_final.haz_type +
                                     str(impact_func_final.id))
    for key, val in param_dict.items():
        df_out[key] = val
    return impact_func_final, df_out

def change_impf(impf_instance, param_dict):
    """apply a shifting or a scaling defined in param_dict to the impact
    function in impf_istance and return it as a new ImpactFunc object.

    Parameters
    ----------
    impf_instance : ImpactFunc
        an instance of ImpactFunc
    param_dict : dict
        dict of parameter_names and values (interpreted as
        factors, 1 = neutral)
        e.g. {'mdd_shift': 1.05, 'mdd_scale': 0.8,
        'paa_shift': 1, paa_scale': 1}

    Returns
    -------
    ImpactFunc : The Impact function based on the parameterisation
    """
    ImpactFunc_new = copy.deepcopy(impf_instance)
    # create higher resolution impact functions (intensity, mdd ,paa)
    paa_func = interpolate.interp1d(ImpactFunc_new.intensity,
                                    ImpactFunc_new.paa,
                                    fill_value='extrapolate')
    mdd_func = interpolate.interp1d(ImpactFunc_new.intensity,
                                    ImpactFunc_new.mdd,
                                    fill_value='extrapolate')
    temp_dict = dict()
    temp_dict['paa_intensity_ext'] = np.linspace(ImpactFunc_new.intensity.min(),
                                                 ImpactFunc_new.intensity.max(),
                                                 (ImpactFunc_new.intensity.shape[0] + 1) * 10 + 1)
    temp_dict['mdd_intensity_ext'] = np.linspace(ImpactFunc_new.intensity.min(),
                                                 ImpactFunc_new.intensity.max(),
                                                 (ImpactFunc_new.intensity.shape[0] + 1) * 10 + 1)
    temp_dict['paa_ext'] = paa_func(temp_dict['paa_intensity_ext'])
    temp_dict['mdd_ext'] = mdd_func(temp_dict['mdd_intensity_ext'])
    # apply changes given in param_dict
    for key, val in param_dict.items():
        field_key, action = key.split('_')
        if action == 'shift':
            shift_absolut = (
                ImpactFunc_new.intensity[np.nonzero(getattr(ImpactFunc_new, field_key))[0][0]]
                * (val - 1))
            temp_dict[field_key + '_intensity_ext'] = \
                temp_dict[field_key + '_intensity_ext'] + shift_absolut
        elif action == 'scale':
            temp_dict[field_key + '_ext'] = \
                    np.clip(temp_dict[field_key + '_ext'] * val,
                            a_min=0,
                            a_max=1)
        else:
            raise AttributeError('keys in param_dict not recognized. Use only:'
                                 'paa_shift, paa_scale, mdd_shift, mdd_scale')

    # map changed, high resolution impact functions back to initial resolution
    ImpactFunc_new.intensity = np.linspace(ImpactFunc_new.intensity.min(),
                                           ImpactFunc_new.intensity.max(),
                                           (ImpactFunc_new.intensity.shape[0] + 1) * 10 + 1)
    paa_func_new = interpolate.interp1d(temp_dict['paa_intensity_ext'],
                                        temp_dict['paa_ext'],
                                        fill_value='extrapolate')
    mdd_func_new = interpolate.interp1d(temp_dict['mdd_intensity_ext'],
                                        temp_dict['mdd_ext'],
                                        fill_value='extrapolate')
    ImpactFunc_new.paa = paa_func_new(ImpactFunc_new.intensity)
    ImpactFunc_new.mdd = mdd_func_new(ImpactFunc_new.intensity)
    return ImpactFunc_new

def init_impact_data(hazard_type,
                     region_ids,
                     year_range,
                     impact_data,
                     reference_year,
                     impact_data_source='emdat',
                     yearly_impact=True):
    """creates a dataframe containing the recorded impact data for one hazard
    type and one area (countries, country or local split)

    Parameters
    ----------
    hazard_type : str
        default = 'TC', type of hazard 'WS','FL' etc.
    region_ids : str
        name the region_ids or country names
    year_range : list
        list containting start and end year.
        e.g. [1980, 2017]
    impact_data : climada.Impact or str
        Climada impact object, path to emdat source file
    reference_year : int
        impacts will be scaled to this year
    impact_data_source : str, optional
        default 'emdat', 'Impact'
    yearly_impact : bool, optional
        if set True, impact is returned per year, not per event

    Returns
    -------
    df_out : pd.DataFrame
        Dataframe with recorded impact written to rows for each year
        or event.
    """
    # if impact_data_source=='Impact':
    #     print('Impact data is assumed to be scaled (inflation adjusted)')
    #     imp_obs_df = pd.DataFrame({'impact_scaled':impact_data.at_event,
    #                             'impact':np.nan,
    #                             'date':impact_data.date})

    #     df_impact_data=pd.DataFrame({'event_id':hazard.event_id,
    #                 'event_name':hazard.event_name,
    #                 'year':[dt.datetime.fromordinal(d).year for d in hazard.date],
    #                 'date':hazard.date})
    #     df_impact_data = df_impact_data.merge(imp_obs_df,how='outer',on='date')     

    if impact_data_source == 'emdat':
        if yearly_impact:
            df_impact_data = emdat_impact_yearlysum(impact_data, countries=region_ids,
                                             hazard=hazard_type,
                                             year_range=year_range,
                                             reference_year=reference_year)
        else:
            raise ValueError('init_impact_data not yet implemented for yearly_impact = False.')
            df_impact_data = emdat_impact_event(impact_data)
    else:
        raise ValueError('init_impact_data not yet implemented for other impact_data_sources '
                         'than emdat.')
    return df_impact_data


def calib_cost_calc(df_out, cost_function):
    """calculate the cost function of the modelled impact impact_CLIMADA and
        the reported impact impact_scaled in df_out

    Parameters
    ----------
    df_out : pd.Dataframe
        DataFrame as created in calib_instance
    cost_function : str
        chooses the cost function e.g. 'R2' or 'logR2'

    Returns
    -------
    cost : float
        The results of the cost function when comparing modelled and
        reported impact
    """
    if cost_function == 'R2':
        cost = np.sum((pd.to_numeric(df_out['impact_scaled']) -
                       pd.to_numeric(df_out['impact_CLIMADA']))**2)
    elif cost_function == 'logR2':
        impact1 = pd.to_numeric(df_out['impact_scaled'])
        impact1[impact1 <= 0] = 1
        impact2 = pd.to_numeric(df_out['impact_CLIMADA'])
        impact2[impact2 <= 0] = 1
        cost = np.sum((np.log(impact1) -
                       np.log(impact2))**2)
    else:
        raise ValueError('This cost function is not implemented.')
    return cost


def calib_all(hazard, exposure, impf_name_or_instance, param_full_dict,
              impact_data_source, year_range,intensity_range, yearly_impact=True):
    """portrait the difference between modelled and reported impacts for all
    impact functions described in param_full_dict and impf_name_or_instance

    Parameters
    ----------
    hazard : list or Hazard
    exposure : list or Exposures
        list or instance of exposure of full countries
    impf_name_or_instance: string or ImpactFunc
        the name of a parameterisation or an instance of class
        ImpactFunc e.g. 'emanuel'
    param_full_dict : dict
        a dict containing keys used for
        f_name_or_instance and values which are iterable (lists)
        e.g. {'v_thresh' : [25.7, 20], 'v_half': [70], 'scale': [1, 0.8]}
    impact_data_source : dict or pd.Dataframe
        with name of impact data source and file location or dataframe
    year_range : list
    intensity_range : np.array
        tuple of 3 intensity numbers along np.arange(min, max, step)
    yearly_impact : bool, optional

    Returns
    -------
    df_result : pd.DataFrame
        df with modelled impact written to rows for each year or event.
    """
    df_result = None  # init return variable

    # prepare hazard and exposure
    region_ids = list(np.unique(exposure.region_id))
    hazard_type = hazard.tag.haz_type
    exposure.assign_centroids(hazard)

    # prepare impact data
    if isinstance(impact_data_source, pd.DataFrame):
        df_impact_data = impact_data_source
    else:
        if list(impact_data_source.keys()) == ['emdat']:
            df_impact_data = init_impact_data(hazard_type, region_ids, year_range,
                                              impact_data_source['emdat'],hazard, year_range[-1])
        else:
            raise ValueError('other impact data sources not yet implemented.')
    params_generator = (dict(zip(param_full_dict, x))
                        for x in itertools.product(*param_full_dict.values()))
    for param_dict in params_generator:
        print(param_dict)
        df_out = copy.deepcopy(df_impact_data)
        impact_func_final, df_out = init_impf(impf_name_or_instance, param_dict,intensity_range, df_out)
        df_out = calib_instance(hazard, exposure, impact_func_final, df_out, yearly_impact)
        if df_result is None:
            df_result = copy.deepcopy(df_out)
        else:
            df_result = df_result.append(df_out, input)


    return df_result


def calib_optimize(hazard, exposure, impf_name_or_instance, param_dict, bounds_dict,
                   impact_data_source, year_range=None, intensity_range=None, yearly_impact=True,
                   cost_function='R2',options=None, show_details=False, insured_fraction = 1):
    """portrait the difference between modelled and reported impacts for all
    impact functions described in param_full_dict and impf_name_or_instance

    Parameters
    ----------
    hazard: list or Hazard
    exposure: list or Exposures
        list or instance of exposure of full countries
    impf_name_or_instance: string or ImpactFunc
        the name of a parameterisation or an instance of class
        ImpactFunc e.g. 'emanuel'
    param_dict : dict
        a dict containing keys used for
        impf_name_or_instance and one set of values
        e.g. {'v_thresh': 25.7, 'v_half': 70, 'scale': 1}
    impact_data_source : dict or pd.DataFrame or climada.engine.Impact
        dict: with name of impact data source and file location or dataframe
        pd.DataFrame: impact data with the following columns:
            event_id: must be identical to hazard
            event_name, date, year, impact_scaled, impact
        climada.engine.Impact: impact data (event_id does not need to match hazard)
    year_range : list
    intensity_range : np.array
        range of hazard intensity. e.g. np.arange(min, max, step)
    yearly_impact : bool, optional
    cost_function : str, optional
        the argument for function calib_cost_calc, default 'R2'
    show_details : bool, optional
        if True, return a tuple with the parameters AND
        the details of the optimization like success,
        status, number of iterations etc
    insured_fraction : float
        if damage data is based on insurance claims, it is possible that not all assets
        in the exposure are actually insured and would appear in the damages. In such a case,
        the fraction of insured assets/value can be specified. The damages are then divided
        by this number to get damage values that are representative for all assets.

    Returns
    -------
    param_dict_result : dict or tuple
        the parameters with the best calibration results
        (or a tuple with (1) the parameters and (2) the optimization output)
    """
    if options is None: options = {'init_points' : 20, 'n_iter' : 80}

    # prepare hazard and exposure
    exposure.assign_centroids(hazard)

    # prepare impact data
    if isinstance(impact_data_source, pd.DataFrame):
        df_impact_data = impact_data_source
        df_impact_data['impact_scaled'] = df_impact_data['impact_scaled']/insured_fraction
    elif isinstance(impact_data_source, Impact):
        if yearly_impact:
            raise NotImplementedError('yearly damages from Imp object not implemented')
        #add impact data to dataframe
        #Exception for non-unique dates
        if len(impact_data_source.date) != len(np.unique(impact_data_source)):
            if all(impact_data_source.event_id == hazard.event_id) and \
                all(impact_data_source.date == hazard.date):
                #if hazard and imp date&id are identical, use hazard event dates
                df_impact_data=pd.DataFrame({'event_id': impact_data_source.event_id,
                            'event_name': impact_data_source.event_name,
                            'year':[dt.datetime.fromordinal(d).year for d in impact_data_source.date],
                            'date': impact_data_source.date,
                            'impact_scaled':impact_data_source.at_event/insured_fraction,
                            'impact':np.nan})
            else:
                raise NotImplementedError('Imp object with multiple events per date not implemented. \
                                          Except hazard and imp date&id are identical.')
        else: #unique dates
            #'impact_scaled' is the relevant columns for calibration, 'impact' will be ignored
            imp_obs_df = pd.DataFrame({'impact_scaled':impact_data_source.at_event/insured_fraction,
                                    'impact':np.nan,
                                    'date':impact_data_source.date})
            #create df_impact_data based on hazard event dates
            df_impact_data=pd.DataFrame({'event_id': hazard.event_id,
                        'event_name': hazard.event_name,
                        'year':[dt.datetime.fromordinal(d).year for d in hazard.date],
                        'date': hazard.date})
            
            #merge impact data into hazard-based dataframe
            df_impact_data = df_impact_data.merge(imp_obs_df,how='outer',on='date')     
    elif isinstance(impact_data_source, dict):
        if list(impact_data_source.keys()) == ['emdat']:
            region_ids = list(np.unique(exposure.region_id))
            hazard_type = hazard.tag.haz_type
            df_impact_data = init_impact_data(hazard_type, region_ids, year_range,
                                              impact_data_source['emdat'], year_range[-1])
        else:
            raise ValueError('other impact data sources not yet implemented.')

    #Warning if yearly_impact=False, but only one impact per year is given
    if (not yearly_impact) and len(df_impact_data.year.unique()) == df_impact_data.shape[0]:
        Warning("Only one recored impact per year. Consider changing yearly_impact to 'True'")
    if yearly_impact: assert(len(df_impact_data.year.unique()) == df_impact_data.shape[0])


    def specific_calib_Bayesian(**param_dict):
        print(param_dict)

        try:
            #initialize impact function
            impf=init_impf(impf_name_or_instance, param_dict,intensity_range)[0]
        except ValueError as e:
            if ' v_half <= v_thresh' in str(e):
                #if invalid input to Emanuel_impf (v_half <= v_thresh), return zero
                return 0
            else:
                raise ValueError(f'Unknown Error in init_impf:{e}. Check inputs!')

        #return 1/cost_function (because BayesianOptimization maximizes the function)
        return 1/calib_instance(hazard, exposure,impf,
                                df_impact_data,
                                yearly_impact=yearly_impact, return_cost=cost_function)

    # Create a BayesianOptimization optimizer,
    # and optimize the given black_box_function.
    optimizer = BayesianOptimization(f = specific_calib_Bayesian,
                                    pbounds = bounds_dict,
                                    verbose = 2,
                                    random_state = 4)
                                    
    optimizer.maximize(**options)#(init_points = 10, n_iter = 30)#
    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

    param_dict_result = optimizer.max["params"]
    assert(param_dict_result.keys()==param_dict.keys())

    if show_details:
        plot_param_space(param_dict_result,optimizer)
        plot_impf(param_dict_result,optimizer,impf_name_or_instance,intensity_range,haz=hazard,
                  err_name = cost_function)
        return param_dict_result, optimizer   

    return param_dict_result


########################################################################################
#non calibration functions
def plot_impf(param_dict_result,optimizer,impf_name_or_instance,intensity_range,haz=None,err_name=''):
    """plot the best impact function and the target function (over time)"""

    target = np.array([i['target'] for i in optimizer.res])
    error = 1/target

    #plot target function (over time)
    fig,ax = plt.subplots()
    ax.plot(error,'o',label=f'error: {err_name}')
    ax.set(yscale = 'log',xlabel='iteration',ylabel=f'error: {err_name}')
    ax.scatter(np.argmin(error),min(error),color='tab:red',zorder=4,label=f'best fit')
    ax.legend()
    #plot best impact function
    fig,ax = plt.subplots()
    title = [f'{key}: {param_dict_result[key]:.2f}' if param_dict_result[key]>0.1 else 
             f'{key}: {param_dict_result[key]:.2e}' for key in param_dict_result.keys()]
    impf = init_impf(impf_name_or_instance,param_dict_result,intensity_range)[0]
    ax.plot(impf.intensity,impf.mdd*impf.paa*100,zorder=3,color='tab:blue',label=f'MDR best fit')#\nerr={min(error):.1e}')
    max_y = max(impf.mdd*impf.paa)*100
    ax.set(ylim=(0,max_y),title=title,xlabel=f'Intensity [{impf.intensity_unit}]',ylabel = 'MDR [%]')

    #plot all impact functions within the given quantile (top 10% by default)
    # q90=np.quantile(target,plot_quant)
    # params_over90 = [i['params'] for i in np.array(a)[target>q90]]
    # error_over90 = [1/i['target'] for i in np.array(a)[target>q90]]

    #Plot all impact functions with an Error less than 10% larger than the best
    params_over90 = [i['params'] for i in np.array(optimizer.res)[error<=min(error)*1.1]]
    error_over90 = [1/i['target'] for i in np.array(optimizer.res)[error<=min(error)*1.1]]    

    for params,error_now in zip(params_over90,error_over90):
        impf = init_impf(impf_name_or_instance,params,intensity_range)[0]
        label = 'within 10 percent of best fit' if error_now==error_over90[0] else None
        ax.plot(impf.intensity,impf.mdd*impf.paa*100,color='grey',alpha=0.3,label=label)
        max_y = max(max_y,max(impf.mdd*impf.paa)*100)
    ax.set(ylim=(0,max_y))
    if haz:
        ax2 = ax.twinx()
        ax2.hist(haz.intensity[haz.intensity.nonzero()].getA1(),bins=40,color='tab:orange',
                 alpha=0.3,label='Haz intensity')
        ax2.set(ylabel='Intensity distribution in hazard data')
        ax2.legend(loc='upper right')
    ax.legend()
    return ax



def plot_param_space(param_dict_result,optimizer):
    """plot the parameter space with the best result highlighted"""

    #get all parameter combinations for 2d plots
    var_combs = list(combinations(param_dict_result.keys(),2))

    #set up figure
    rows = np.ceil(len(param_dict_result.keys())/3).astype(int)
    cols = min(len(param_dict_result.keys()),3)
    fig,axes = plt.subplots(rows,cols,figsize=(4.5*cols,3.5*rows),gridspec_kw={'wspace':0.3})

    #scatter plot each parameter combination
    for i,var_comb in enumerate(var_combs):
        ax = axes.flatten()[i]
        idx1=list(param_dict_result.keys()).index(var_comb[0])
        idx2=list(param_dict_result.keys()).index(var_comb[1])
        # fig,ax = plt.subplots()
        scat=ax.scatter(optimizer.space.params[:,idx1],optimizer.space.params[:,idx2],c=optimizer.space.target)
        ax.scatter(optimizer.max['params'][var_comb[0]],optimizer.max['params'][var_comb[1]],marker='x',color='red')
        if 'v_thresh' in var_comb and 'v_half' in var_comb:
            ax.plot([0,100],[0,100],color='grey')
        ax.set(xlabel=var_comb[0],ylabel=var_comb[1],title=var_comb)
        # ax.get_xaxis().set_major_formatter(StrMethodFormatter('{x:.1e}'))#(plt.LogFormatter(10,  labelOnlyBase=False))

    #set colorbar
    cbar_ax = fig.add_axes([0.25, -0.1, 0.5, 0.06]) #left,bot,width,height
    fig.colorbar(scat, cax=cbar_ax,orientation='horizontal').set_label('Inverse Error measure')


def get_emanuel_impf(v_thresh=20, v_half=60, scale=1e-3,power=3,
                    impf_id=1, intensity=np.arange(0, 110, 1), 
                    intensity_unit='mm',haz_type='HL'):
    """
    Init TC impact function using the formula of Kerry Emanuel, 2011:
    'Global Warming Effects on U.S. Hurricane Damage',
    https://doi.org/10.1175/WCAS-D-11-00007.1

    Parameters
    ----------
    impf_id : int, optional
        impact function id. Default: 1
    intensity : np.array, optional
        intensity array in m/s. Default:
        5 m/s step array from 0 to 120m/s
    v_thresh : float, optional
        first shape parameter, wind speed in
        m/s below which there is no damage. Default: 25.7(Emanuel 2011)
    v_half : float, optional
        second shape parameter, wind speed in m/s
        at which 50% of max. damage is expected. Default:
        v_threshold + 49 m/s (mean value of Sealy & Strobl 2017)
    scale : float, optional
        scale parameter, linear scaling of MDD.
        0<=scale<=1. Default: 1.0
    power : int, optional
        Exponential dependence. Default to 3 (as in Emanuel (2011))

    Raises
    ------
    ValueError

    Returns
    -------
    impf : ImpfTropCyclone
        TC impact function instance based on formula by Emanuel (2011)
    """
    if v_half <= v_thresh:
        raise ValueError('Shape parameters out of range: v_half <= v_thresh.')
    if v_thresh < 0 or v_half < 0:
        raise ValueError('Negative shape parameter.')
    if scale > 1 or scale <= 0:
        raise ValueError('Scale parameter out of range.')

    impf = ImpactFunc(haz_type=haz_type, id=impf_id,intensity=intensity,
                        intensity_unit=intensity_unit,name='Emanuel-type')
    impf.paa = np.ones(intensity.shape)
    v_temp = (impf.intensity - v_thresh) / (v_half - v_thresh)
    v_temp[v_temp < 0] = 0
    impf.mdd = v_temp**power / (1 + v_temp**power)
    impf.mdd *= scale
    return impf


def fit_emanuel_impf_to_emp_data(emp_df,pbounds,opt_var='MDR',options=None,optimizer='Nelder-Mead'):

    if not emp_df.index.name:
        raise ValueError('Careful, emp_df.index has no name. Double check if the \
                            index corresponds to the intensity unit!')
    
    def weighted_inverse_MSE(**param_dict):
        try:
            impf = get_emanuel_impf(**param_dict,intensity=emp_df.index.values)
        except ValueError as e:
            if 'v_half <= v_thresh' in str(e):
                #if invalid input to Emanuel_impf (v_half <= v_thresh), return zero
                return 0
            else:
                raise ValueError(f'Unknown Error in init_impf:{e}. Check inputs!')
        if opt_var == 'MDR':
            mdr = impf.mdd*impf.paa
            SE = np.square(mdr-emp_df.MDR)
        elif opt_var == 'PAA':
            paa = impf.mdd*impf.paa #PAA is used implicitly, output MDD will be PAA
            SE = np.square(paa-emp_df.PAA)
        elif opt_var == 'MDD':
            mdd = impf.mdd*impf.paa #MDD is used implicitly, output MDD will be MDD
            SE = np.square(mdd-emp_df.MDD)

        SE_weigh = SE*emp_df.count_cell
        SE_weigh_noZero = SE_weigh[emp_df.index.values!=0]
        MSE = np.mean(SE_weigh_noZero)
        return 1/MSE

    if optimizer == 'Bayesian':
        if options is None: options = {'init_points' : 20, 'n_iter' : 80}
        optimizer = BayesianOptimization(f = weighted_inverse_MSE,
                                        pbounds = pbounds,
                                        verbose = 2,
                                        random_state = 4)
                                        
        optimizer.maximize(**options)#(init_points = 10, n_iter = 30)#
        print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

        param_dict_result = optimizer.max["params"]
        # assert(param_dict_result.keys()==param_dict.keys())
        plot_param_space(param_dict_result,optimizer)
        impf = init_impf('emanuel_HL',param_dict_result,emp_df.index.values)[0]
        ax=plot_impf(param_dict_result,optimizer,'emanuel_HL',emp_df.index.values)

    elif optimizer == 'Nelder-Mead':
        param_means = [(v[0]+v[1])/2 for v in pbounds.values()]
        param_dict = dict(zip(pbounds.keys(),param_means))
        bounds = [(v[0],v[1]) for v in pbounds.values()]
        x0 = list(param_dict.values())

        #define function that return the negative of 1/MSE -> proportional to MSE
        def neg_mse(x):
            param_dict_temp = dict(zip(param_dict.keys(), x))
            return -weighted_inverse_MSE(**param_dict_temp)
        
        res = minimize(neg_mse, x0,
                        bounds=bounds,
                        #bounds=((0.0, np.inf), (0.0, np.inf), (0.0, 1.0)),
                        # constraints=cons,
                        # method='SLSQP',
                        method = 'Nelder-Mead',
                        # method='trust-constr',
                    options={'xtol': 1e-5, 'disp': True, 'maxiter': 500})

        optimizer = res
        param_dict_result = dict(zip(param_dict.keys(), res.x))
        print(param_dict_result)
        impf = init_impf('emanuel_HL',param_dict_result,emp_df.index.values)[0]
        ax=impf.plot(zorder=3)
        title = [f'{key}: {param_dict_result[key]:.2f}' if param_dict_result[key]>0.1 else f'{key}: {param_dict_result[key]:.2e}' for key in param_dict_result.keys()]
        ax.set(ylim=(0,max(impf.mdd*100)),title=title)

    #add empirical function to plot
    ax.plot(emp_df.index,emp_df[opt_var]*100,label=f'Empirical {opt_var}')
    plt.legend()

    return param_dict_result, optimizer, impf

# if __name__ == "__main__":
#
#
#    ## tryout calib_all
#    hazard = TropCyclone.from_hdf5('C:/Users/ThomasRoosli/tc_NA_hazard.hdf5')
#    exposure = LitPop.from_hdf5('C:/Users/ThomasRoosli/DOM_LitPop.hdf5')
#    impf_name_or_instance = 'emanuel'
#    param_full_dict = {'v_thresh': [25.7, 20], 'v_half': [70], 'scale': [1, 0.8]}
#
#    impact_data_source = {'emdat':('D:/Documents_DATA/EM-DAT/'
#                                   '20181031_disaster_list_all_non-technological/'
#                                   'ThomasRoosli_2018-10-31.csv')}
#    year_range = [2004, 2017]
#    yearly_impact = True
#    df_result = calib_all(hazard,exposure,impf_name_or_instance,param_full_dict,
#                  impact_data_source, year_range, yearly_impact)
#
#
#    ## tryout calib_optimize
#    hazard = TropCyclone.from_hdf5('C:/Users/ThomasRoosli/tc_NA_hazard.hdf5')
#    exposure = LitPop.from_hdf5('C:/Users/ThomasRoosli/DOM_LitPop.hdf5')
#    impf_name_or_instance = 'emanuel'
#    param_dict = {'v_thresh': 25.7, 'v_half': 70, 'scale': 0.6}
#    year_range = [2004, 2017]
#    cost_function = 'R2'
#    show_details = True
#    yearly_impact = True
#    impact_data_source = {'emdat':('D:/Documents_DATA/EM-DAT/'
#                                   '20181031_disaster_list_all_non-technological/'
#                                   'ThomasRoosli_2018-10-31.csv')}
#    param_result,result = calib_optimize(hazard,exposure,impf_name_or_instance,param_dict,
#              impact_data_source, year_range, yearly_impact=yearly_impact,
#              cost_fucntion=cost_function,show_details= show_details)
#
#
