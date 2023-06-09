"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.18.3
"""


from kedro.io import DataCatalog
from kedro.extras.datasets.pandas import CSVDataSet
data_catalog = DataCatalog({'heart_failure_prepared': CSVDataSet(filepath='data/03_prepared/heart_failure_prepared.csv')})

def dropUnusedCollumns(df):

    df = df.drop('sex', axis=1)
    return df

def repairData(df):

    data_catalog.save('heart_failure_prepared', df)
    return df.fillna(value=0)

