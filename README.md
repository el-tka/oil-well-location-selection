# Oil Well Location Selection

Machine learning project for selecting the most profitable region for oil well drilling based on geological exploration data.

---

## Project Overview

Oil exploration companies must decide where to drill new wells while minimizing financial risks. Each drilling operation requires significant investment, so selecting the most profitable region is critical.

The goal of this project is to build a machine learning model that predicts the volume of oil reserves in potential wells and to determine which region provides the highest expected profit with acceptable risk.

The project combines **machine learning, economic modeling, and risk analysis** to support data-driven decision making in oil exploration.

---

## Dataset

The dataset contains geological exploration data for **three regions**.
Each region includes information for **10,000 potential wells**.

Features represent geological characteristics of wells.

Target variable:

**product** — estimated oil reserves in the well.

---

## Methodology

The analysis consists of several stages.

### 1. Data Preparation

* loading datasets
* feature selection
* splitting data into training and validation sets

### 2. Model Training

A **Linear Regression** model was used to predict oil reserves based on geological features.

Model evaluation metric:

**RMSE (Root Mean Squared Error)**

### 3. Profit Calculation

Using model predictions, the expected profit was calculated for each region considering:

* drilling budget
* number of wells selected for development
* revenue per unit of oil

### 4. Risk Analysis

To evaluate financial risks, **Bootstrap simulation** was used.

This allowed:

* simulation of profit distribution
* estimation of confidence intervals
* calculation of probability of losses

---

## Results

The analysis identified the region with:

* the **highest expected profit**
* an **acceptable probability of losses (< 2.5%)**

This approach allows companies to optimize investment decisions when selecting drilling locations.

---

## Technologies

Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Bootstrap simulation
Jupyter Notebook

---

## Project Structure

```
oil-well-location-selection
│
├── data
│   ├── geo_data_0.csv
│   ├── geo_data_1.csv
│   └── geo_data_2.csv
│
├── notebooks
│   └── oil_well_analysis.ipynb
├── utils
│   └── oil_well_analysis.html
│
├── src
│   ├── preprocessing.py
│   ├── model.py
│   └── bootstrap.py
│    └── pipeline.py
│
├── images
│   ├── profit_distribution.png
│   └── bootstrap_results.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Repository Contents

**data/**
Raw datasets used in the analysis.

**notebooks/**
Jupyter notebook containing exploratory analysis and experiment workflow.

**src/**
Python modules with reusable code:

* `preprocessing.py` — data preparation
* `model.py` — machine learning model training
* `bootstrap.py` — profit simulation and risk analysis
* `pipeline.py` — starting script

**images/**
Visualizations used in the project documentation.

---

## Key Skills Demonstrated

* exploratory data analysis (EDA)
* regression modeling
* economic modeling of oil production
* bootstrap risk simulation
* business decision support with machine learning

---

## Business Impact

The developed approach helps oil companies:

* identify the most profitable drilling region
* reduce investment risks
* support strategic exploration decisions using data analysis.
