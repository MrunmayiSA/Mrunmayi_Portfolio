# Mrunmayi Anchawale Portfolio

### Project 1: Churn Prediction of bank customers
* Created a model that predicts whether a customer will churn (his/her balance will fall below minimum balance in the next quarter)
* Handled missing values using appropriate imputations
* Pre-processed data to make it modelling ready (one-hot encoding, skeweness treatment, outlier removal, scaling)
* Created baseline model to serve as an indicator of performance for other models (along with its confusion matrix, ROC-AUC on test data), with cross-validation
* Compared baseline against model having all features
* Compared baseline agaist model built using top 10 features obtained by Reverse Feature Elimination

[Project Implementation](https://github.com/MrunmayiSA/CustomerChurnPrediction.git)

### Project 2: Analysis of US Traffic Accidents and Classification into Severity levels
* Dataset: https://smoosavi.org/datasets/us_accidents
* Described summary statistics of the dataset and perfomed exploratory data analysis
* Utilized sampling techniques in order to :
  * Reduce size of dataset (3 million accident records)
  * Overcome class imbalance (130k severe accidents, 270k non-severe accidents)
* Applied logistic regression and using AIC as performance measure, dropped a few features
* Calculated sensitivity, specificity for model
* Used lasso regression, decision tree, random forest to improve model performance

[Conclusion and Results](https://zhuochenglin.github.io/US_Accidents_Project/)
