import json

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from scipy.differentiate import derivative
from scipy.special import expit
from scipy.stats import uniform

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb

#the goal of this project is to predict whether a transaction is fraudulent or not.
#in an unbalanced dataset we might use focal loss, undersampling, oversampling, epoch manipulation, anomaly detection
#I will try each solution and see which one works best for this dataset
#in addition to those data techniques, I will also try different models such as logistic regression, random forest and xgboost
#for each model I will try different hyperparameters and thresholds to see which one works best for this dataset

#we record the results in a json file, and then we can analyze the results to see which combination works best for this dataset.


#note that: fraud is positive and non-fraud is negative

def prepare_data(data):

    #the class time needs special treatment, it's the number of seconds elapsed between this transaction and the first transaction in the dataset.
    #we want to turn it into a feature that represents the time of day, so we can use it to detect patterns in the data.
    data['Time'] = data['Time'] % (24 * 60 * 60) #convert seconds to seconds in a day
    data['Minute_of_day'] = data['Time'] / 60 #convert seconds to minutes

    y = data['Class']
    X = data.drop(['Class'], axis=1) #drop the target variable from the features

    return X, y

def RENN(X, y):
    print('number of negative samples before undersampling:', {sum(y==0)})
    renn = RepeatedEditedNearestNeighbours()
    X_resampled, y_resampled = renn.fit_resample(X, y)
    print('number of negative samples after undersampling:', {sum(y_resampled==0)})
    return X_resampled, y_resampled

def random_undersampling(X, y):
    print('number of negative samples before undersampling:', {sum(y==0)})
    rus = RandomUnderSampler(sampling_strategy=0.01, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print('number of negative samples after undersampling:', {sum(y_resampled==0)})
    return X_resampled, y_resampled


def SMOTE_oversampling(X, y):
    print('number of positive samples before oversampling:', {sum(y==1)})
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print('number of positive samples after oversampling:', {sum(y_resampled==1)})
    return X_resampled, y_resampled


def improve_data(X, y):
    #undersampling
    # X, y = RENN(X, y)
    X, y = random_undersampling(X, y)


    #oversampling
    X, y = SMOTE_oversampling(X, y)

    log = {
        'data_improvement': {
            'under_sampling': 'Random Under Sampling',
            'over_sampling': 'SMOTE',
        }
    }

    return X, y, log


def fit_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    #store results in the log file
    log = {
        'model': 'Logistic Regression',
        'hyperparameters': {
            'max_iter': 1000,
        }
    }

    return model, log


def fit_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #store results in the log file
    log = {
        'model': 'Random Forest',
        'hyperparameters': {
            'n_estimators': 100,
        }
    }

    return model, log


def focal_loss_obj(preds, dtrain, gamma=1.0, alpha=0.1):
    """
    Custom Focal Loss objective for XGBoost.
    """
    labels = dtrain.get_label()
    p = expit(preds)  # Convert log-odds to probability

    # Calculate Gradient
    # grad = -alpha * (1-p)^gamma * (gamma*p*log(p) + p - label) ... simplified:
    grad = p - labels
    grad = (alpha * (labels) * (1 - p) ** gamma * grad +
            (1 - alpha) * (1 - labels) * p ** gamma * grad)

    # Calculate Hessian (Second derivative)
    # For simplicity, many practitioners use a simplified version
    # or the standard Hessian of Cross Entropy if gamma is small.
    # Below is the approximate Hessian for Focal Loss:
    hess = p * (1 - p)
    hess = (alpha * (labels) * (1 - p) ** gamma * hess +
            (1 - alpha) * (1 - labels) * p ** gamma * hess)

    return grad, hess

def fit_xgboost_with_focal_loss(X_train, y_train):
    params = {
        'max_depth': 6,
        'eta': 0.1,
        'eval_metric': 'auc'
    }
    model = xgb.train(params, obj=focal_loss_obj, dtrain=xgb.DMatrix(X_train, label=y_train), num_boost_round=100)
    log = {
        'model': 'XGBoost with Focal Loss',
        'hyperparameters': {
            'num_boost_round': 100,
            'gamma': 2.0,
            'alpha': 0.1
        }
    }
    return model, log

def fit_xgboost(X_train, y_train):
    model = XGBClassifier(n_estimators=100, random_state=42, objective=focal_loss_obj)
    model.fit(X_train, y_train)

    # random_search = RandomizedSearchCV(estimator=model, param_distributions={
    #     'learning_rate': uniform(0.05, 0.3), #search for learning rates between 0.01 and 0.3
    #     'n_estimators': [100, 200, 300, 400, 500]
    # }, n_iter=10, cv=3, verbose=2, random_state=42, scoring='f1')
    # random_search.fit(X_train, y_train)

    #store results in the log file
    log = {
        'model': 'XGBoost',
        # 'hyperparameters': {
            # 'n_estimators': random_search.best_params_['n_estimators'],
            # 'learning_rate': random_search.best_params_['learning_rate'],
        # }
    }

    return model, log


def fit_LightGBM(X_train, y_train):
    model = LGBMClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #store results in the log file
    log = {
        'model': 'LightGBM',
        'hyperparameters': {
            'n_estimators': 100,
        }
    }

    return model, log



#try anomaly detection using isolation forest, and see if it improves the results
# def fit_isolation_forest(X_train, y_train):
#     model = IsolationForest(contamination=0.01, random_state=42)
#     model.fit(X_train)
#
#     #store results in the log file
#     log = {
#         'model': 'Isolation Forest',
#         'hyperparameters': {
#             'contamination': 0.01,
#         }
#     }
#
#     return model, log

def evaluate_model(model, X_val, y_val):
    X_val = xgb.DMatrix(X_val) #for focal loss specifically
    raw_preds = model.predict(X_val)
    probs = expit(raw_preds)  # Use sigmoid to get 0-1 range
    y_pred = [1 if p > 0.4 else 0 for p in probs]
    # y_pred = model.predict(X_val)
    # probs = model.predict_proba(X_val)[:, 1] #get the probabilities of the positive class
    # threshold = 0.9 #adjusting this threshold to see how it affects the results
    # y_pred = (probs >= threshold).astype(int) #convert probabilities to binary

    f2 = fbeta_score(y_val, y_pred, beta=2)
    f1 = fbeta_score(y_val, y_pred, beta=1)
    f_1_2 = fbeta_score(y_val, y_pred, beta=1/2)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)

    #store results in the log file
    log = {
        'threshold': 0.5,
        'evaluation': {
            'F2 Score': f2,
            'F1 Score': f1,
            'F1/2 Score': f_1_2,
            'False Positive Rate': fpr,
            'precision': precision,
            'recall': recall,
            'specificity': specificity
        }
    }

    return log

if __name__ == "__main__":
    #load the dataset
    train = pd.read_csv('split/train.csv')
    val = pd.read_csv('split/val.csv')

    total_log = {}

    #prepare data
    X_train, y_train = prepare_data(train)
    X_val, y_val = prepare_data(val)

    scaler = MinMaxScaler()
    X_val = scaler.fit_transform(X_val)

    #imporve data using over and/or under sampling and/or more
    X_train, y_train, log = improve_data(X_train, y_train)
    total_log = total_log | log

    X_train = scaler.fit_transform(X_train)

    #fit the model
    # model, log = fit_logistic_regression(X_train, y_train)
    # model, log = fit_random_forest(X_train, y_train)
    # model, log = fit_xgboost(X_train, y_train)
    # model, log = fit_LightGBM(X_train, y_train)
    model, log = fit_xgboost_with_focal_loss(X_train, y_train)
    total_log = total_log | log #append the log of the model to the total log

    #evaluate the model on the validation set
    log = evaluate_model(model, X_val, y_val)
    total_log = total_log | log

    with open('results_log.jsonl', 'a') as log_file:
        #write the total log to the log file
        log_file.write(json.dumps(total_log, indent=4) + '\n')

    print("Results have been logged to results_log.jsonl")