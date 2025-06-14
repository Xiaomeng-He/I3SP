# I3SP

## Introduction

This repository contains the code and implementation details for the paper **Inter-Case Informed Business Process Suffix Prediction Integrating Trace and Log Information**.

To facilitate navigation, below is an overview of the files:

- **`1_create_prefix_suffix/`**  
  Contains scripts for data preprocessing, dataset splitting, and generating trace prefixes, log prefixes, and trace suffixes:  
  - `preprocessing.py`: Handles data preprocessing
  - `train_test_split.py`: Handles dataset splitting.  
  - `create_prefix_suffix.py`: Generates trace prefixes, log prefixes, and trace suffixes.  

- **`2_Seq2Seq/`**  
  Contains scripts for creating trace-based and integrated Seq2Seq models, each available in two variants: with or without the attention mechanism:
  - `create_Seq2Seq.py`: Implements the Seq2Seq model.
  - `train_evaluate_Seq2Seq.py`: Defines loss functions and performance metrics to train and evaluate the Seq2Seq model.

- **`3_SEP_LSTM/`**  
  Contains scripts for creating both trace-based and integrated SEP-LSTM models:  
  - `create_SEP_LSTM.py`: Implements the SEP-LSTM model.
  - `train_evaluate_SEP_LSTM.py`: Defines loss functions to train the SEP-LSTM model, as well as performance metrics to evaluate the iteratively generated suffix.

- **`4_SEP_XGBoost/`**  
  Contains scripts for creating both trace-based and integrated SEP-XGBoost models:  
  - `SEP_XGBoost_pipeline.ipynb`: Trains two separate XGBoost models—one for predicting the next activity label and another for the next timestamp. It then iteratively generates suffixes using the trained models and evaluates their performance.

## Python Environment Setup

The research project is implemented using Python 3.12.7. To install all required packages, download `requirements.txt` and run the following command:

```bash
pip install -r requirements.txt
