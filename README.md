# Adult Income Dataset - Data Cleaning and Preparation (Q1b)

## Overview

This repository contains the cleaned version of the UCI Adult Income dataset, along with a Jupyter Notebook demonstrating the data cleaning and preparation pipeline. The work was done as part of a Business Intelligence (BI) workflow assignment, focusing on Q1(b): systematic data cleaning and preparation.

The dataset and cleaning process are intended to support downstream BI analysis, credit scoring, and income classification modeling.

---

## Files in this Repository

- `data/adult.data` : Raw UCI Adult dataset.
- `adult_cleaned.csv` : Cleaned dataset after the data preparation pipeline.
- `Q1b_cleaning.ipynb` : Jupyter Notebook containing the step-by-step data cleaning code.

---

## Cleaning and Preparation Pipeline

The data cleaning and preparation pipeline was implemented using **Python (pandas)** and **Jupyter Notebook** to ensure **full reproducibility**. The main steps included:

1. **Standardisation of categorical labels**
   - Converted all text columns to lowercase and stripped extra spaces to ensure consistency.
2. **Imputation of missing values**
   - Replaced missing values (`?`) with `'unknown'` for categorical variables to preserve all rows.
3. **Removal of redundant variables**
   - Dropped `education` since it is redundant with `education_num`.
4. **Encoding of categorical attributes**
   - Applied one-hot encoding (with `drop_first=True`) for all categorical features to make the data suitable for BI analysis and modeling.
5. **Target variable preparation**
   - Mapped `income` to binary values (`<=50k → 0`, `>50k → 1`) for modeling purposes.

The pipeline ensures that anyone can **reproduce the cleaned dataset** by running the notebook.
