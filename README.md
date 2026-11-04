# Comparison of Machine Learning Models (Car Evaluation)

A comparison of four classification models (Random Forest, Gradient Boosting, XGBoost and LightGBM) on the UCI Car Evaluation dataset, built with Python and scikit-learn.

## Overview
The notebook `Makine Öğrenmesi Modellerinin Karşılaştırılması.ipynb` predicts the acceptability class of a car (`unacc`, `acc`, `good`, `vgood`) from its price, maintenance cost, number of doors, passenger capacity, luggage boot size and safety rating. The goal is to compare how the models perform, how stable they are, and which features they rely on.

## Dataset
- Source: UCI Car Evaluation (1,728 records, 6 features, 4 classes)
- No missing values and no duplicate rows
- Imbalanced classes: unacc ~70%, acc ~22%, good ~4%, vgood ~4%
- Categorical features encoded with `OrdinalEncoder` (low → high order)

## Workflow
1. Data loading, duplicate and missing value checks
2. Ordinal encoding of categorical features
3. Exploratory analysis (class distribution, correlation matrix)
4. Stratified 80/20 train/test split
5. Training of 4 models with class balancing (`class_weight` / `sample_weight`)
6. Evaluation with classification reports and feature importance plots
7. 5-Fold Cross-Validation comparison
8. Example prediction for a new car

## Models
- Random Forest (bagging)
- Gradient Boosting Machine (scikit-learn)
- XGBoost
- LightGBM

## Results

| Metric | Random Forest | GBM | XGBoost | LightGBM |
|--------|---------------|-----|---------|----------|
| Test Accuracy | 0.9249 | 0.9942 | 0.9913 | 0.9827 |
| CV Accuracy (mean) | 0.8420 | 0.8438 | 0.8496 | 0.8542 |

**Key findings**
- GBM has the highest test accuracy but a large gap to its CV score (~0.15), which points to overfitting. A single train/test split alone is not a reliable metric.
- LightGBM has the best CV accuracy (85.4%) with low variance (±0.026), so it is the most stable and generalizable model here.
- Safety and passenger capacity are the most influential features in most models, which matches the correlation analysis.

## Tools
Python, pandas, scikit-learn, XGBoost, LightGBM, matplotlib, seaborn, Jupyter Notebook

## How to Run
```bash
git clone https://github.com/Ahmet4254/Comprehension-Of-Machine-Learning-Models.git
cd Comprehension-Of-Machine-Learning-Models
pip install pandas scikit-learn xgboost lightgbm matplotlib seaborn requests jupyter
jupyter notebook "Makine Öğrenmesi Modellerinin Karşılaştırılması.ipynb"
```
The dataset is downloaded automatically from a public GitHub URL inside the notebook.
