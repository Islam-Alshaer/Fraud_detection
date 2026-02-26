import json
import pandas as pd
import models
import xgboost as xgb
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix


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




def evaluate_score(y_pred, y_val):
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
        'threshold': 0.45,
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


def evaluate_model(model, X_val, y_val):
    # y_pred = model.predict(X_val)


    X_val = xgb.DMatrix(X_val) #for focal loss specifically
    raw_preds = model.predict(X_val)
    probs = expit(raw_preds)  # Use sigmoid to get 0-1 range
    y_pred = [1 if p > 0.45 else 0 for p in probs]

    # probs = model.predict_proba(X_val)[:, 1] #get the probabilities of the positive class
    # threshold = 0.3 #adjusting this threshold to see how it affects the results
    # y_pred = (probs >= threshold).astype(int) #convert probabilities to binary
    # # y_pred = [1 if i==-1 else 0 for i in y_pred] #for isolation forest specifically, we need to convert the predictions to binary

    # # #for vote models
    # y_pred = model[0].predict(X_val)
    # for i in range(1, len(model)):
    #     sub_model = model[i]
    #     y_pred = y_pred + sub_model.predict(X_val)
    #
    # y_pred = (y_pred / len(model) >= 0.5).astype(int) #average the predictions and convert to binary

    return evaluate_score(y_pred, y_val)


if __name__ == "__main__":
    #load the dataset
    train = pd.read_csv('split/train.csv')
    # val = pd.read_csv('split/val.csv')
    val = pd.read_csv('split/test.csv')
    total_log = {}

    #prepare data
    X_train, y_train = prepare_data(train)
    X_val, y_val = prepare_data(val)

    scaler = MinMaxScaler()
    X_val = scaler.fit_transform(X_val)

    #imporve data using over and/or under sampling and/or more
    X_train, y_train, log = improve_data(X_train, y_train)
    # log = {}
    total_log = total_log | log
    X_train = scaler.fit_transform(X_train)

    #fit the model
    # model, log = models.fit_logistic_regression(X_train, y_train)
    # model, log = models.fit_random_forest(X_train, y_train)
    # model, log = models.fit_xgboost(X_train, y_train)
    # model, log = models.fit_LightGBM(X_train, y_train)
    model, log = models.fit_xgboost_with_focal_loss(X_train, y_train)
    # model, log = models.fit_logistic_regression_with_class_weight(X_train, y_train)
    # model, log = models.fit_isolation_forest(X_train, y_train)
    # model, log = models.fit_KNN(X_train, y_train)
    # model, log = models.fit_kmeans(X_train, y_train)
    # model, log = models.fit_vote(X_train, y_train)

    total_log = total_log | log #append the log of the model to the total log

    #evaluate the model on the validation set
    log = evaluate_model(model, X_val, y_val)
    total_log = total_log | log

    with open('results_log.jsonl', 'a') as log_file:
        #write the total log to the log file
        log_file.write(json.dumps(total_log, indent=4) + '\n')

    print("Results have been logged to results_log.jsonl")



    '''
    results for test set: 
    
{
    "data_improvement": {
        "under_sampling": "Random Under Sampling",
        "over_sampling": "SMOTE"
    },
    "model": "XGBoost with Focal Loss",
    "hyperparameters": {
        "num_boost_round": 100,
        "gamma": 4.0,
        "alpha": 0.6
    },
    "threshold": 0.45,
    "evaluation": {
        "F2 Score": 0.7410562180579217,
        "F1 Score": 0.5878378378378378,
        "F1/2 Score": 0.48712206047032475,
        "False Positive Rate": 0.0019696463429646695,
        "precision": 0.4371859296482412,
        "recall": 0.8969072164948454,
        "specificity": 0.9980303536570353
    }
}
 
    '''