import pandas as pd
import numpy as np

import random
import pickle

from sklearn import metrics



# Define feature names
static_feature_names = ['Sex', 'BW', 'SGA', 'Multiple', 'hCAM' ,'PPROM','Oligo','Maternal DM', 'Maternal HTN','Maternal age > 30', '5min AS < 6', 'C/S', 'GA']
VentFiO2_feature_names = ['area FiO2', 'area vent']
BGA_feature_names = ["area pH", "area CO2"]
static_BGA_feature_names = static_feature_names + BGA_feature_names
static_VentFiO2_feature_names = static_feature_names + VentFiO2_feature_names
all_feature_names = static_feature_names + BGA_feature_names + VentFiO2_feature_names

# Build Random Data
data = pd.DataFrame(np.random.uniform(0, 1, size=(1000, len(all_feature_names))), columns=all_feature_names) # Generate 1000 random samlpes
data['MS-BPD'] =[random.randint(0,1) for x in range(1000)]

y_true = data['MS-BPD']

def evaluate(y_true, y_pred):
    auprc = metrics.average_precision_score(y_true, y_pred)
    auroc = metrics.roc_auc_score(y_true, y_pred)
    return auroc, auprc

# Static Model
filename = 'models/static/model0.pkl'
with open(filename, 'rb') as inp:
    model = pickle.load(inp)    
y_pred_static = model.predict_proba(data[static_feature_names])[:,1]

# VentFiO2
filename = 'models/VentFiO2/model0.pkl'
with open(filename, 'rb') as inp:
    model = pickle.load(inp)    
y_pred_VentFiO2 = model.predict_proba(data[VentFiO2_feature_names])[:,1]

# BGA
filename = 'models/BGA/model0.pkl'
with open(filename, 'rb') as inp:
    model = pickle.load(inp)    
y_pred_BGA = model.predict_proba(data[BGA_feature_names])[:,1]

# Static + BGA
filename = 'models/static_BGA/model0.pkl'
with open(filename, 'rb') as inp:
    model = pickle.load(inp)    
y_pred_sBGA = model.predict_proba(data[static_BGA_feature_names])[:,1]

# Static + VentFiO2
filename = 'models/static_VentFiO2/model0.pkl'
with open(filename, 'rb') as inp:
    model = pickle.load(inp)    
y_pred_sVentFiO2 = model.predict_proba(data[static_VentFiO2_feature_names])[:,1]

# All
filename = 'models/all/model0.pkl'
with open(filename, 'rb') as inp:
    model = pickle.load(inp)
y_pred_all = model.predict_proba(data[all_feature_names])[:,1]

auroc, auprc = evaluate(y_true, y_pred_sVentFiO2)
print("AUROC: {}, AUPRC: {}".format(auroc, auprc))
