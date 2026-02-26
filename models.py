from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import xgboost as xgb
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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
    model = XGBClassifier(n_estimators=100, random_state=42)
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
def fit_isolation_forest(X_train, y_train):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)

    #store results in the log file
    log = {
        'model': 'Isolation Forest',
        'hyperparameters': {
            'contamination': 0.1,
        }
    }

    return model, log

def fit_KNN(X_train, y_train):
    model = KNeighborsClassifier()

    #use grid search to find the best n_neighbors
    grid_search = GridSearchCV(estimator=model, param_grid={
        'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10]
    }, scoring='f1')
    grid_search.fit(X_train, y_train)
    # model.fit(X_train, y_train)

    log = {
        'model': 'K-Nearest Neighbors',
        'hyperparameters': {
            'n_neighbors': grid_search.best_params_['n_neighbors'],
        }
    }

    return grid_search, log

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

def fit_logistic_regression_with_class_weight(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    #store results in the log file
    log = {
        'model': 'Logistic Regression with Class Weight',
        'hyperparameters': {
            'max_iter': 1000,
            'class_weight': 'balanced'
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


def focal_loss_obj(preds, dtrain, gamma=4.0, alpha=0.6):
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
            'gamma': 4.0,
            'alpha': 0.6
        }
    }
    return model, log

def fit_xgboost(X_train, y_train):
    model = XGBClassifier(n_estimators=100, random_state=42)
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
def fit_isolation_forest(X_train, y_train):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)

    #store results in the log file
    log = {
        'model': 'Isolation Forest',
        'hyperparameters': {
            'contamination': 0.1,
        }
    }

    return model, log

def fit_KNN(X_train, y_train):
    model = KNeighborsClassifier()

    #use grid search to find the best n_neighbors
    grid_search = GridSearchCV(estimator=model, param_grid={
        'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10]
    }, scoring='f1')
    grid_search.fit(X_train, y_train)
    # model.fit(X_train, y_train)

    log = {
        'model': 'K-Nearest Neighbors',
        'hyperparameters': {
            'n_neighbors': grid_search.best_params_['n_neighbors'],
        }
    }

    return grid_search, log


def fit_kmeans(X_train, y_train):

    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X_train)

    log = {
        'model': 'K-Means Clustering',
    }

    return model, log

def fit_vote(X_train, y_train):
    #fit the three models and use majority voting to make predictions
    model1 = fit_LightGBM(X_train, y_train)[0]
    # model2 = fit_random_forest(X_train, y_train)[0]
    model2 = fit_logistic_regression_with_class_weight(X_train, y_train)[0]
    model3 = fit_xgboost(X_train, y_train)[0]

    log = {
        'model': 'Voting Classifier',
        'hyperparameters': {
            'models': ['LightGBM', 'Logistic regression with class weight', 'XGBoost']
        }
    }

    return [model1, model2, model3], log