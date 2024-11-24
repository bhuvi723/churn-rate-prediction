# 📊 Churn Rate Prediction

Welcome to the Churn Rate Prediction using telco dataset project! This repository contains all the necessary files and instructions to understand and run the project. 

## 📋 About

This project aims to predict customer churn using machine learning & deep learning techniques. Customer churn refers to the loss of clients or customers. By predicting churn, businesses can take proactive measures to retain customers and improve their services. The project includes data preprocessing, model training, evaluation, and deployment using Streamlit.

## 📁 Repository Structure

- `customer_churn.csv` 📂: Contains the dataset used for training and testing the model.
- `churn-rate-prediction.py` 📓: Jupyter notebooks with exploratory data analysis, model building & training, optuna hyper parameter tuning.
- `new_model.pkl` 📁: This file consists best saved ANN model after tuing the model with optuna.
- `iframe_figures/` 📁: This file consists images of optuna analysis visualizations.
- `main.py` 🌐: Streamlit app python file for deploying the model.
- `requirements.txt` 📄: List of dependencies required to run the project.
- `README.md` 📄: This file, providing an overview of the project.

## 🚀 Getting Started

### Cloning the Repository

To clone the repository, use the following command:

```bash
git clone https://github.com/yourusername/churn-rate-pred.git
cd churn-rate-pred
```

## 🛠️ Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### Running the Streamlit App

For running the Streamlit app:

```bash
streamlit run main.py
```

The Streamlit app is deployed on Render. To access the deployed app, visit the following link:

[Streamlit App on Render](https://telcom-churn-rate-prediction.onrender.com)

Happy Predicting! 🎉