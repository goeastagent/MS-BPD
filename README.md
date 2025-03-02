# MS-BPD Prediction Model

This repository contains a predictive model for Moderate-to-Severe Bronchopulmonary Dysplasia (MS-BPD) based on one week of time-series data and static clinical factors. The model utilizes **Random Forest** as the classifier to predict the risk of MS-BPD in preterm infants.

## Features Used
We incorporate both static clinical factors and time-series physiological data for model development:

### Static Factors:
- Sex
- Birth weight
- Small for gestational age (SGA)
- Histological chorioamnionitis
- Maternal diabetes mellitus
- Maternal hypertension
- Apgar score
- Cesarean section (C-section)
- Preterm premature rupture of membranes (PPROM)

### Time-Series Factors (one-week window):
- Blood Gas Analysis (BGA): pH, pCO2
- Ventilation and Oxygen Support: FiO2, Ventilation settings

## Models Provided
To evaluate the impact of different feature sets, we provide six different models:

1. Static Only Model: Uses only static clinical factors.
2. BGA Model: Uses only pH and pCO2.
3. Ventilation and FiO2 Model (VentFiO2): Uses only ventilation settings and FiO2.
4. Static + BGA Model: Combines static factors with pH and pCO2.
5. Static + VentFiO2 Model: Combines static factors with ventilation settings and FiO2.
6. All Features Model: Uses all available features (static factors + BGA + VentFiO2).

## Model Implementation
The models are implemented using **Random Forest**, a robust ensemble learning method that handles both static and time-series features effectively. The dataset is preprocessed to extract relevant feature representations before training.

## Evaluation
Model performance is assessed using standard classification metrics:
- AUROC (Area Under the Receiver Operating Characteristic Curve)
- AUPRC (Area Under the Precision-Recall Curve)

## Repository Structure
```
├── models/             # Saved trained models
├── example_code.py     # Example script
├── README.md           # Project documentation
```
