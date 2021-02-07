# Repository for Kaggle Competitions

This repo contains code solutions and configurations for entered Kaggle competitions. The project is divided up into the following three subdirectories:

1. [Competitions](#competitions)
    1. [Digit Recognizer](#digit-recognizer)
	2. [House Prices](#house-prices-advanced-regression-techniques)
	3. [Predict Future Sales](#competitive-data-science)
	4. [Titanic](#titanic)
2. [Environments](#environments)
3. [Utilties](#utilities)

## Competitions <a name="competitions"></a>

### Digit Recognizer <a name="digit-recognizer"></a>

This subdirectory contains the code for the [digit recogniser competition](https://www.kaggle.com/c/digit-recognizer). The goal of the competition is to classify tens of thousands of handwritten images from the MNIST ("Modified National Institute of Standards and Technology") dataset. The simple dataset is often used many computer vision tasks and serves as a basis for benchmarking classification algorithms.

### House Prices <a name="house-prices-advanced-regression-techniques"></a>

The HousePrices directory holds all of the code for the [House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The purpose of this competition is to predict the final price of residential homes in Ames, Iowa using 79 explanatory variables such as basement ceiling height and distance to railroad. The data has plenty of scope for feature engineering and advanced regression techniques.

### Predict Future Sales <a name="competitive-data-science"></a>

The code contained within this folder is responable for the [Predict Future Sales competition](https://www.kaggle.com/c/competitive-data-science-predict-future-sales). This challenge serves as the final project for the Coursera course, ["How to win a data science competition"](https://www.coursera.org/learn/competitive-data-science). The competition uses a challenging time-series dataset of daily sales data proved by [1C Company](https://1c.ru/eng/title.htm), one of the largest Russian software firms. The goal of the competition is to predict the total sales for every product and store in the next month.

### Titanic <a name="titanic"></a>

The Titanic folder contains the code for the [Titanic - Machine Learning from Disaster competition](https://www.kaggle.com/c/titanic). The aim of the competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck. The dataset used in the competition holds passenger information such as name, age, gender and soci-economic class.

## Environments <a name="environments"></a>

The environments subdirectory contains .yml files for installing previously configured conda environments for competition use.

## Utilities <a name="utilities"></a>

This folder contains utility functions for interacting with the Kaggle Platform through Python using the [Kaggle API](https://github.com/Kaggle/kaggle-api). These utility functions are applicable to all competitions and can be used for:

- Downloading the relevant competition data from Kaggle 
- Submitting competition predictions for evaluation on the Kaggle Platform
- Returning the results and performance of previous competition submissions.