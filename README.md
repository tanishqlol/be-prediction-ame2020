# Nuclear Binding Energy Prediction Model

## Overview
This project implements a **stacked ensemble model** to predict nuclear binding energies using data from the AME2020 mass evaluation.

## Model Architecture

The model uses a two-stage approach:

1. **Neural Network Ensemble (Stage 1)**
   - 50 feedforward neural networks trained via Monte Carlo cross-validation
   - Architecture: [3 → 64 → 64 → 1] with batch normalization and dropout
   - Each model trained on different random train/val splits
   - Predictions averaged to reduce variance

2. **XGBoost Residual Correction (Stage 2)**
   - Trains on residuals from the NN ensemble
   - Captures systematic errors missed by neural networks
   - Final prediction = NN ensemble + XGBoost correction

## Features
The model uses only three basic nuclear properties:
- **Z**: Number of protons
- **N**: Number of neutrons  
- **A**: Mass number (A = Z + N)

## Results

**Final Test Performance:**
- **RMSE: 0.094 MeV**
- **MAE: 0.048 MeV**

See `exec.ipynb` for the further results and plots

## Data
- Source: AME2020 mass evaluation (`mass_1.mas20.txt`)
- ~3,000 experimentally measured nuclei
- Split: 72% train / 18% validation / 10% test
