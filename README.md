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

### Project 3: Prediction of trip duration for cab services
* Eliminated irrelevant variables. Explored target variable. 
* Visulized features and drew conclusions (Univariate, bivariate analysis)
* Found feature relationships with target
* Performed pre-processing and feature engineering
* After train-test split, constructed regression model with 5-fold cross validation using linear regression and decision trees
* Evaluated models using MSE

[Project Implementation](https://github.com/MrunmayiSA/PredictingTripDurationRegression.git)

### Project 4: Named Entity Recognition model
* Task: To capture adverse drug reaction of a patient to a particular drug documented in patient's electronic health record using NLP
* Explored Information Retrieval techniques for formulating rules to extract word sequences which form possible adverse drug reactions.
* Utilized sci-spacy's pre-trained models for identification of diseases since they can be considered as adverse drug reactions.
* Fine-tuned the NER model developed at John Snow Labs, using SparkNLP pipeline

[Project Implementation](https://github.com/MrunmayiSA/NERModel.git)

### Project 5: Clustering on 2D array of points
* Performed k-means clustering while visualizing clusters at each iteration
* Used elbow method to find the optimal value of k
* Implemented dbscan, hierarchical clustering, while visualizing dendograms
* Performed color quantization using k-means

[Project Implementation](https://github.com/MrunmayiSA/ClusteringOn2DArray.git)

### Project 6: Dimensionality Reduction
* Perfomed PCA on lfw_people dataset with 10 principal components
* Found number of components required to preserve 95% variance in the dataset (k)
* Updated PCA model with k and reconstructed the faces
* Trained logistic regression and KNN classifier on images
* Evaluated model performance using f-1 score

[Project Implementation](https://github.com/MrunmayiSA/DimensionalityReductionOnFaces.git)
