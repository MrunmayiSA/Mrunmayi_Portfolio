# Mrunmayi Anchawale Portfolio

### Project 1: Fake Job Classifier
* Dataset : Employment Scam Aegean Dataset (http://emscad.samos.aegean.gr/)
* Performed exploratory data analysis to find the words used to make the jobs sound genuine and industries that attract scammers
* Processed text data (stopwords, punctuation removal) and one-hot-encoded it for ML
* Used Synthetic Minority Oversampling Technique to overcome class imbalance in the target (17,014 real and 866 fake job postings)
* Used Grid Search Cross Validation to find the best parameters for ML models (10 folds)
* Following classification algorithms were tested and compared:
  * Logistic(Ridge) regression
  * KNN (22 neighbours)
  * SVM (rbf kernel)
  * Random Forest (100 estimators)
  * MLP (relu activation function, adam solver)
* Both Random Forest and MLP gave best performance with an ROC-AUC score of 0.99

[Project Implementation](https://github.com/MrunmayiSA/FakeJobClassifier.git)

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

### Project 3: Music Genre Classification
Dataset: “Musical genre classification of audio signals “ by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002
* Extracting following features from spectogram:
  * Mel-frequency cepstral coefficients (MFCC)(20 in number)
  * Spectral Centroid,
  * Zero Crossing Rate
  * Chroma Frequencies
  * Spectral Roll-off
* Encoding the Labels
* Scaling the Feature columns
* Classification with Keras

[Project Implementation](https://github.com/MrunmayiSA/MusicGenreClassification.git)

### Project 4: Book Recommendation System
Coming Soon........




### Project 5: Text Summarization
* Researched current advances in the field of NLP for summarization
* Explored inner-workings of PEGASUS model and using it for transfer learning
* Fine-tuned BART and T5 models on the CNN/Daily Mail dataset 
* Tested these models on BBC News dataset and compared them with each other as well as the baseline model using ROUGE score<br>

|  Models   |rouge1 |rouge2 |rougeL |prec1  |prec2  | precL |
|-----------|-------|-------|-------|-------|-------|-------|
|Baseline-3 | 43.88 | 36.04 | 34.75 | 77.81 | 63.95 | 61.71 |
|Baseline-1 | 22.45 | 19.57 | 21.66 | 87.03 | 77.93 | 83.70 |
|BART-Base  | 5.40  | 0.37  | 4.36  | 29.52 | 22.6  | 24.08 |
|BART-Tuned | 2.58  | 0.18  | 2.11  | 15.23 | 1.18  | 12.56 |
|T5-Small   | 5.30  | 0.38  | 4.384 | 33.69 | 2.73  | 28.35 |
|T5-Tuned   | 1.51  | 0.09  | 1.31  | 11.29 | 0.81  | 9.98  |

[Project Implementation](https://github.com/MrunmayiSA/Text-Summarization.git)

### Project 6: Named Entity Recognition model
* Task: To capture adverse drug reaction of a patient to a particular drug documented in patient's electronic health record using NLP
* Explored Information Retrieval techniques for formulating rules to extract word sequences which form possible adverse drug reactions.
* Utilized sci-spacy's pre-trained models for identification of diseases since they can be considered as adverse drug reactions.
* Fine-tuned the NER model developed at John Snow Labs, using SparkNLP pipeline

[Project Implementation](https://github.com/MrunmayiSA/NERModel.git)

### Project 7: Predict Electricity Consumption - ARIMA
* Dataset: https://www.kaggle.com/kandij/electric-production
* Visualized Time Series
* Checked for stationarity using ADF (Augmented Dickey-Fuller) Test
* Removed trends by moving averages and seasonality by differencing
* Constructed ACF and PACF plots to get the values of p and q
* Fitted ARIMA model with RSS of 0.52 and predicted consumption for the period 2017-2024

[Project Implementation](https://github.com/MrunmayiSA/ElectricityConsumptionPrediction.git)


### Project 8: Dimensionality Reduction
* Perfomed PCA on lfw_people dataset with 10 principal components
* Found number of components required to preserve 95% variance in the dataset (k)
* Updated PCA model with k and reconstructed the faces
* Trained logistic regression and KNN classifier on images
* Evaluated model performance using f-1 score

[Project Implementation](https://github.com/MrunmayiSA/DimensionalityReductionOnFaces.git)

### Project 9: Clustering on 2D array of points
* Performed k-means clustering while visualizing clusters at each iteration
* Used elbow method to find the optimal value of k
* Implemented dbscan, hierarchical clustering, while visualizing dendograms
* Performed color quantization using k-means

[Project Implementation](https://github.com/MrunmayiSA/ClusteringOn2DArray.git)


### Project 10: Churn Prediction of bank customers
* Created a model that predicts whether a customer will churn (his/her balance will fall below minimum balance in the next quarter)
* Handled missing values using appropriate imputations
* Pre-processed data to make it modelling ready (one-hot encoding, skeweness treatment, outlier removal, scaling)
* Created baseline model to serve as an indicator of performance for other models (along with its confusion matrix, ROC-AUC on test data), with cross-validation
* Compared baseline against model having all features
* Compared baseline agaist model built using top 10 features obtained by Reverse Feature Elimination

[Project Implementation](https://github.com/MrunmayiSA/CustomerChurnPrediction.git)


### Project 11: Prediction of trip duration for cab services
* Eliminated irrelevant variables. Explored target variable. 
* Visulized features and drew conclusions (Univariate, bivariate analysis)
* Found feature relationships with target
* Performed pre-processing and feature engineering
* After train-test split, constructed regression model with 5-fold cross validation using linear regression and decision trees
* Evaluated models using MSE

[Project Implementation](https://github.com/MrunmayiSA/PredictingTripDurationRegression.git)







