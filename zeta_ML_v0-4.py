# -*- coding: utf-8 -*-
"""
Created on Sat Apr 3 11:27:09 2021

@author: Jonathan Machin
Zeta-potential prediction model v0-4
Last update: 20/12/2021
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb


# set global plot parameters
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.dpi'] = 300


# data dictionaries for lipid phosphate charge, headgroup charge and Tm
lip_charge  = {'DMPC': -1, 'DPPC': -1, 'DOPC': -1, 'POPG': -1, 'DSPC': -1, 'POPC': -1, 'DMPG': -1,
               'Chol': 0, 'DOTAP': 0, 'LP': 0, 'DOPS': -1, 'TOCL': -2, 'POPE': -1,
               'DOPE': -1, 'DMPE': -1, 'DC-Chol': 0, 'EPC': 0, 'DSPE': -1, 'DMPA': -1, 'DMPS': -1, 'SM': -1}

head_charge = {'DMPC': 1, 'DPPC': 1, 'DOPC': 1, 'POPG': 0, 'DSPC': 1, 'POPC': 1, 'DMPG': 0,
               'Chol': 0, 'DOTAP': 1, 'LP': 0, 'DOPS': 0, 'TOCL': 0, 'POPE': 1,
               'DOPE': 1, 'DMPE': 1, 'DC-Chol': 1 , 'EPC': 1, 'DSPE': 1, 'DMPA': 0, 'DMPS': 0, 'SM': 1}

# NOTE: EPC/LP Tm is not known (these are removed prior to processing)
# TOCL phase transition estimated
tm = {'DMPC': 24, 'DPPC': 41, 'DOPC': -17, 'POPG': -2, 'DSPC': 55, 'POPC': -2, 'DMPG': 23,
      'Chol': 0, 'DOTAP': -5, 'LP': None, 'DOPS': -11, 'TOCL': -5, 'POPE': 25,
      'DOPE': -16, 'DMPE': 50, 'DC-Chol': 0 , 'EPC': None, 'POPE': 25, 'DSPE': 74, 'DMPA': 52, 'DMPS': 35, 'SM': None,}

# ------------ CURVE FITTING FUNCTIONS ------------
# cubic fit
def cubic(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d

# quadratic fit
def quadratic(x,a,b,c):
    return a*x**2 + b*x + c

fitting_functions = {'cubic':cubic, 'quadratic':quadratic}
# ----------- END CURVE FITTING FUNCTIONS ------------



# read the dataset file and generate general lipid parameters
def parse_lipids(zd, mean_sd=None):
    return_msd = False
    lip_charge_sum = []
    head_charge_sum = []
    overall_charge_sum = []
    tm_sum = []
    chol_sum = []
   
    
    if mean_sd == None:
        mean_sd = zd['sd'].mean()
        return_msd = True
    for i in zd['sd']:
        if not isinstance(i, float):
            print(i, 'is not numeric')
    zd.fillna(value=mean_sd, inplace=True)
    
    for i in zd['lipid']:
        split = i.split(' ')
        if len(split) == 1:
            if 'Chol' in split[0]:
                chol = 100
            else:
                chol = 0
            lipc = lip_charge[split[0]]
            headc = head_charge[split[0]]
            if 'Chol' not in split[0]:
                tmc = tm[split[0]]
                
        elif len(split) == 2:
            l = split[0].split('/')
            r = [float(j) for j in split[1].split(':')]
    
            lipc_total, headc_total, tmc_total, chol_total = 0, 0, 0, 0
            for ind, lip in enumerate(l):
                if 'Chol' in lip:
                    chol_total += r[ind]
                
                lipc_total += lip_charge[lip] * r[ind]
                headc_total += head_charge[lip] * r[ind]
                
                if 'Chol' not in lip:
                    tmc_total += tm[lip] * r[ind]
                
            
            lipc = lipc_total / sum(r)
            headc = headc_total / sum(r)
            tmc = tmc_total / sum(r)
            chol = chol_total / sum(r)
            
        else:
            print('incorrect lipid-information format, this MAY break the script')
    
        lip_charge_sum.append(lipc)
        head_charge_sum.append(headc)
        overall_charge_sum.append(lipc/2+headc/2)
        tm_sum.append(tmc)
        chol_sum.append(chol)
               
    zd['lip_charge'] = lip_charge_sum
    zd['head_charge'] = head_charge_sum
    zd['overall_charge'] = overall_charge_sum
    zd['tm'] = tm_sum
    zd['chol'] = chol_sum
    
    if return_msd == True:
        return zd, mean_sd
    else:
        return zd

# handles initial dataframe generation from data file    
def process_data(file):
    zd = pd.read_csv(file)    
    
    # remove lipids that not enough information are known about
    # EPC/LP/PC/SM lipids removed as acyl chains not stated in the paper (or mix of unknown ratio used)
    # therefore Tm cannot be determined
    zd = zd[~zd.lipid.str.contains("EPC")]
    zd = zd[~zd.lipid.str.contains("LP")]
    zd = zd[zd.lipid != "PC"]
    zd = zd[zd.lipid != "SM"]
    
    # removed if ethanol in the buffer
    zd = zd[~zd.notes.str.contains("ethanol")]
    
    # removed if KClO4 is ion in buffer (rather than KCl/NaCl)
    zd = zd[~zd.notes.str.contains("KClO4")]
    
    zd, mean_sd = parse_lipids(zd)

    return zd, mean_sd

   

# train a cross_validated model using xgboost
def cv_boost(norm=True, random_split=False, cv=True, weight_scale=5):
    # normalise data in-column, recommended
    if norm==True:
        zd_norm = zd[features]
        zd_norm = (zd_norm-zd_norm.min())/(zd_norm.max()-zd_norm.min())
    else:
        zd_norm = zd
    
    # invert sd for weighting (i.e. so small sd are given a larger weighting)
    weight_lower = 0.5-((1/weight_scale)/2)
    zd_norm['sd'] = 1 - zd_norm['sd']
    zd_norm['sd'] = weight_lower + (zd_norm['sd']/weight_scale)
    
    # generate training data features
    target = ['zeta']
    x, y = zd_norm[features], zd_norm[target]
    data_dmatrix = xgb.DMatrix(data=x[train_features],label=y, weight = zd_norm['sd'])
    
    # set randomness for train-test split
    # only set to False for debugging
    if random_split==False:
        random_state_num = 100
    else:
        random_state_num = random.randint(0,1000000)
        print('using random state seed of', random_state_num, 'for model ensemble generation')
    
    # split the data for train-test, and make sd weights
    train_weight_x, val_weight_x, train_y, val_y = train_test_split(x, y, random_state=random_state_num, test_size=0.25)
    train_x = train_weight_x[train_features]
    weights = train_weight_x['sd']
    
    val_x = val_weight_x[train_features]
    weight_val_x = val_weight_x['sd']
    
    # set xgboost parameters
    params = {"learning_rate":0.05, "n_jobs":1, "booster":'gbtree',
                             "min_child_weight":2.5,
                             "seed":100,
                             "max_depth":6,
                             "subsample":0.85, "colsample_bytree":0.85}
    
    # if doing cross-validation (not prediction) set up the cv process
    if cv==True:
        cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=4,
                        num_boost_round=2500,early_stopping_rounds=25,metrics="mae", 
                        as_pandas=True, seed=100)
        
        if norm==True:
            cv_results[cv_results.keys()] = (cv_results[cv_results.keys()] * (zd.zeta.max()-zd.zeta.min())) 

        print(cv_results)
        
    # define model
    model = xgb.XGBRegressor(n_estimators=2500,learning_rate=0.05, n_jobs=1, booster='gbtree',
                             min_child_weight=2.5, # helps reduce overfitting
                             seed=100,
                             max_depth=6,
                             gamma=0, subsample=0.85, colsample_bytree=0.85,
                             #alpha=0.2
                                 )
    # train model to data
    model.fit(train_x, train_y, verbose=False, 
              early_stopping_rounds=25, eval_set=[(val_x, val_y)],
              sample_weight=weights)

    # predict using model
    predictions = model.predict(val_x)
    
    # if data was normalised, then unnormalise for output, and calculate model MAE
    if norm==True:
        unnorm_preds = (predictions*(zd.zeta.max()-zd.zeta.min())) + zd.zeta.min()
        unnorm_val_y = (val_y*(zd.zeta.max()-zd.zeta.min())) + zd.zeta.min()  
        mae = mean_absolute_error(unnorm_preds, unnorm_val_y)
    else:
        mae = mean_absolute_error(predictions, val_y)
        
    print('MAE of model:', mae)

    return mae, model

    
# combine model importance (gain/weight) per feature for model ensemble, and plot 
def combine_importance(imp, plot=False):
    feat = train_features
    
    summary = {}
    sd = {}
    for f in feat:
        summary[f] = 0
        sd[f] = []
    for im in imp:
        for f in feat:
            summary[f] += (im[f]/len(imp))
            sd[f].append(im[f])
    
    if plot == True:
        name, value, stdev = [],[],[]
        for n,v in summary.items():
            name.append(n)
            value.append(v)
            stdev.append(np.std(sd[n]))
            
        
        name = ['Salt\n(monovalent)', 'Salt\n(divalent)', 'pH', 'Size', 'Temperature', 'Overall\nlipid charge', 'Overall\nlipid Tm', 'Cholesterol']
        fig = plt.figure(figsize=(12,5))
        plt.rcParams['xtick.labelsize'] = 13
        plt.bar(name, value, color='black', zorder=3)
        plt.grid(c=(0.9,0.9,0.9,0.9), zorder=0)
        plt.errorbar(name,value,yerr=stdev, c='black', capsize=4, fmt='none')
        plt.ylabel('Feature weight', fontsize=18)
        plt.ylim(0)
        plt.show()
    
    print('feature importance average:', summary)
    return summary
            

# predict using the generated model ensemble
# returns the unnormalised predictions. Also writes them out to file
def predict(model):
    headings = ['lipid', 'salt conc (mono)', 'salt conc (di)', 'pH', 'Rh', 'temp','sd']
    data = []
    lip, saltmo, saltdi, ph, rh, temp = prediction_parameters
    lip = lip.strip()
    
    for i in x_pred:
        a = str(lip)+' '+str(i)+':'+str(101-i)
        d = [a]+[saltmo, saltdi, ph, rh, temp, np.NaN]
        data.append(d)
        
     
    df = pd.DataFrame(data, columns=headings)
    df = parse_lipids(df, mean_sd)
    
    zd_n = zd[train_features]
    
    df_norm = df[train_features]
    df_norm = (df_norm-zd_n.min())/(zd_n.max()-zd_n.min())


    predictions = model.predict(df_norm)
    unnorm_preds = (predictions*(zd.zeta.max()-zd.zeta.min())) + zd.zeta.min()
    
    
    
    ax1.scatter(x_pred, unnorm_preds, alpha=0.4, s=12, marker='x')
    ax1.grid(c=(0.9,0.9,0.9,0.9))
    ax1.set_axisbelow(True)
     
    return unnorm_preds



# fit curve to the predicted data
def fit_prediction(predictions, curve='cubic'):
    fit = fitting_functions[curve]
    print(predictions)
    popt, pcov = curve_fit(fit, np.array(x_pred), np.array(predictions))
    print('Fitted parameters to', curve, ':', popt)
    ax1.plot(x_pred, fit(np.array(x_pred), *popt), color='black', lw=2) 
        
    return popt


# handles main running of the prediction
def run(mode, importance_plot=False, fit=False, fit_curve='cubic'):   
    # lists to store trained models and their parameters
    ensemble = []
    weight = []
    gain = []
    predictions = []
    maes = []
    
        
    # cv and test parameters
    cv, test = [1, True, False, True, 4], [50, True, True, False, 4]
    
    # set general script parameters based on mode
    if mode == 'cv':
        s = cv
    elif mode == 'predict':
        s = test
        
    
    # find stdev weight scaling
    ws = s[4]
    weight_lower = 0.5-((1/ws)/2)
    print('using data standard deviation as weighting with scale:', ws)
    print('lower, upper bound for weighting:',weight_lower, 1-weight_lower)
    
    while len(ensemble) < s[0]:
        mae, model = cv_boost(norm=s[1], random_split=s[2], cv=s[3], weight_scale=s[4])
        if mae < 5:
            maes.append(mae)
            ensemble.append(model)
            gain.append(model.get_booster().get_score(importance_type='gain'))
            weight.append(model.get_booster().get_score(importance_type='weight'))
    
            predictions.append(predict(model))
            
    
        else:
            print('model mae is considered too high (>5), retraining')
            
    print('average MAE: ', sum(maes)/s[0])
    if len(ensemble) > 1:
        avg = [round((sum(col))/len(col),5) for col in zip(*predictions)]
        ax1.scatter(x_pred, avg, alpha=1, s=20, marker='o', c='black')
        ax1.set_ylabel('\u03B6-potential (mV)', fontsize=16)
        ax1.set_xlabel('% '+str(prediction_parameters[0].split('/')[0]), fontsize=16)
        # ax1.set_title(str(prediction_parameters[0]), fontsize=20)
    else:
        avg = [round((sum(col))/len(col),5) for col in zip(*predictions)] 
    
    
    if fit == True:
        fit_prediction(avg, curve=fit_curve)
        
    cwd = os.getcwd()
    cwd =  cwd.replace('\\', '/') + '/'
    with open(cwd+'prediction.csv', 'w') as out:
        out.write('lipid'+','+'prediction\n')
        for ind, i in enumerate(x_pred):
            out.write(str(i)+', '+str(avg[ind])+'\n')
    
    
    # find model weight and gain
    if importance_plot == True:
        weight_summary = combine_importance(weight, plot=True)
        gain_summary = combine_importance(gain, plot=True)

    

# features for training and inference
features = ['salt conc (mono)', 'salt conc (di)', 'pH', 'Rh', 'temp', 'overall_charge', 'tm', 'chol', 'zeta', 'sd']
train_features = ['salt conc (mono)', 'salt conc (di)', 'pH', 'Rh', 'temp', 'overall_charge', 'tm', 'chol'] 

# general axis and prediction range, used by multiple functions
fig, ax1 = plt.subplots(1,1, figsize=(8,5))
x_pred = range(0,101,1)

# Lipid pair to predict, tuple with the following data:
# 'lipid pair', 'salt conc (mono)', 'salt conc (di)', 'pH', 'Rh', 'temp'
# eg: prediction_parameters = ('DMPS/DMPC', 100, 0, 8.5, 60, 25)
# while theoretically this may work with more than bi-lipid mixes this has not been implemented

prediction_parameters = ('DMPE/DMPG', 100, 0, 8.5, 60, 25)
       


# read in dataset file as pandas dataframe
file = "C:/Users/jmmac/Desktop/project/scripts/raw_zeta_data.csv"
zd, mean_sd = process_data(file)


# set mode (cv: find cross-validation of single model, 
#           predict: predict using OTF generated model ensemble)
run(mode='predict', importance_plot=True, fit=False, fit_curve='cubic')









