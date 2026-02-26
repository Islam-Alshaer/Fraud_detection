# Introduction
this project is about detecting fraud transactions using a tabular dataset of over **170,000**
transactions. We use 30 features, 28 of them are anonymous and 2 are known.
# Challenge
the nature of this data is challenging because it’s imbalanced, such that positive class (fruad) is
**0.172%** of the data with only 492 frauds in the data, which makes models tend to predict everything as
negative as it will lead to less loss. In addition, *accuracy metric here is misleading*, as predicting
everything naively as negative (non-fraud) will lead to accuracy of 99.7% despite missing all of the
positive class.
# Requirements
In a real life scenario, it will be up to management to decide which metric matters most. Also,
the amount of false positives acceptable is to be given by the customer. In this project I *assume that
recall and FPR are the most important metrics of all*, because we need to detect frauds without
annoying customers with a lot of false alarms. **I assume 0% - 3% is the range of acceptable FPR for the
benefit of recall in range 75-100%**. Those numbers are assumed based on personal view of the project,
however we can study it more if we want to see the economical and legal effect of recall and FPR.
# Approach
In this project I tried the following techniques for handling imbalanced datasets:
## for data:
1. random under sampling
2. repeated edited nearest neighbors under sampling
3. SMOTE oversampling
## for models:
1. weighted class logistic regression
2. focal loss with XGBoost
3. random forest (as tree based models tend to perform better with poor balanced datasets)
4. vote based models

of course more than that were tried, but those were the most significant. It’s important to notice
that we have to get the best threshold and hyper parameters for each model of those.
# Results
analysis of the results given by the validation data are documented in the file
results_analysis.ipynb.
# The final suggestions are:
1. random undersampling with SMOTE oversampling with XGBoost with focal loss, with a recall
of 0.86 and FPR of 0.005 and precision of 0.18. , so in a real life scenario,
86% of frauds are detected and 5 out of 1000 transactions are false alarms, which is -allegedly-
acceptable in a real life scenario.
2. random undersampling with SMOTE oversampling with weighted logistic regression
as it has recall of 0.87 and FPR of 0.009 and a precision of 0.12.
3. if we really care about precision, we may use XGBoost with no over or under sampling,
with precision of 0.91 and FPR of 0.0001, but recall is only 0.73.
# Test set
Usinig XGBoost with focal loss, alpha: 0.6, gamma: 4.0, threshold: 0.45, The final scores are:

recall: 0.89

FPR: 0.001

precision: 0.43

F2 score: 0.741

F1 score: 0.58



