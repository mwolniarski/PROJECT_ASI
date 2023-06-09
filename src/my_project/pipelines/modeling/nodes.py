"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.18.3
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from prometheus_client import start_http_server, Gauge
import wandb
import logging
import mlflow
import time

# Initialize Prometheus metrics
start_http_server(8000)
roc_auc_metric = Gauge('wandb_roc_auc', 'ROC AUC')
accuracy_metric = Gauge('wandb_accuracy', 'Accuracy metric from Wandb')
def prepare_data_for_modeling(df):
    # Suppress "a copy of slice from a DataFrame is being made" warning
    pd.options.mode.chained_assignment = None

    features = df.columns[1:-1]
    x = df[features]
    y = df['DEATH_EVENT']

    #create a dataframe with the features and the labels
    data_prepared = pd.concat([x, y], axis=1)
    return data_prepared

def split_data(df):

    features = df.columns[1:-1]
    x = df[features]
    y = df["DEATH_EVENT"]
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)
    
    return x_train, x_test, y_train, y_test
def train_model(X_train, y_train):    
    pd.options.mode.chained_assignment = None

    wandb.init(project="Heart failure", mode='offline', name='heart_failure')
    model = LogisticRegression(solver="liblinear")
    model.fit(X_train, y_train)

    return model
def evaluate_model(model, X_test, y_test):
    labels = y_test.unique()
    y_pred = model.predict(X_test)
    y_probas = model.predict_proba(X_test)[:, 1]

    table=wandb.Table(data=X_test, columns=X_test.columns)
    wandb.log({"X_test": table})

    # evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probas)

    raw_data = wandb.Artifact('raw_data', type='dataset')
    raw_data.add_dir('data/01_raw')
    wandb.log_artifact(raw_data)
    training_dataset = wandb.Artifact('training_dataset', type='dataset')

    training_dataset.add_file('data/03_prepared/heart_failure_prepared.csv')
    wandb.log_artifact(training_dataset)
    model_artifact = wandb.Artifact('model', type='model')
    model_artifact.add_file('data/04_model/model.pkl')
    mlflow.log_artifact('data/04_model/model.pkl')
    mlflow.log_artifact('data/03_prepared/heart_failure_prepared.csv')
    wandb.log_artifact(model_artifact)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc auc", roc_auc)
    roc_auc_metric.set(roc_auc)
    accuracy_metric.set(accuracy)
    wandb.log({"ROC_AUC": roc_auc})
    wandb.log({"Accuracy": accuracy})
    print('ROC AUC: %.3f' % roc_auc)
    print('Accuracy: %.3f' % accuracy)

    logger = logging.getLogger(__name__)
    logger.info("Model has an accuracy of %.3f on test data.", accuracy)
    logger.info("Model has an ROC AUC of %.3f on test data.", roc_auc)

    # Delay to allow Prometheus to scrape the W&B logs
    time.sleep(5)