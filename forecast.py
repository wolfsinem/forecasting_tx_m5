import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from tqdm.notebook import tqdm as tqdm
import statsmodels.api as sm

# !pip install fbprophet
# from fbprophet import Prophet
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

# The two paths below were cleaned (see other notebooks)
# train_path =  '../input/cleanedm5data/cleaned_train_data (1).csv'
# test_path = '../input/cleanedm5data/cleaned_test_data (1).csv'
# sample_path = '../input/afcs2021/sample_submission_afcs2021.csv'

# Importing all of the files
calendar = pd.read_csv('../input/afcs2021/calendar_afcs2021.csv')
selling_prices = pd.read_csv('../input/afcs2021/sell_prices_afcs2021.csv')
sample_submission = pd.read_csv('../input/afcs2021/sample_submission_afcs2021.csv')
sales_train_val = pd.read_csv('../input/afcs2021/sales_train_validation_afcs2021.csv')
sales_test = pd.read_csv('../input/afcs2021/sales_test_validation_afcs2021.csv')


ids = sorted(list(set(sales_train_val['id'])))
d_cols = [c for c in sales_train_val.columns if 'd_' in c]

# Create a small training and validation sets to train and validate our models.
# Only use the last 30 days sales as the validation data and the sales of the
# 70 days before that as the training data.
train_dataset = sales_train_val[d_cols[-100:-30]]
val_dataset = sales_train_val[d_cols[-30:]]

# Naive
predictions = []
for i in range(len(val_dataset.columns)):
    if i == 0:
        predictions.append(train_dataset[train_dataset.columns[-1]].values)
    else:
        predictions.append(val_dataset[val_dataset.columns[i - 1]].values)

predictions = np.transpose(np.array([row.tolist() for row in predictions]))
error_naive = np.linalg.norm(predictions[:3] - val_dataset.values[:3]) / len(predictions[0])

# Moving Average
predictions = []
for i in range(len(val_dataset.columns)):
    if i == 0:
        predictions.append(np.mean(train_dataset[train_dataset.columns[-30:]].values, axis=1))
    if i < 31 and i > 0:
        predictions.append(0.5 * (np.mean(train_dataset[train_dataset.columns[-30 + i:]].values, axis=1) + \
                                  np.mean(predictions[:i], axis=0)))
    if i > 31:
        predictions.append(np.mean([predictions[:i]], axis=1))

predictions = np.transpose(np.array([row.tolist() for row in predictions]))
error_avg = np.linalg.norm(predictions[:3] - val_dataset.values[:3]) / len(predictions[0])

# Holt Linear
predictions = []
for row in tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
    fit = Holt(row).fit(smoothing_level = 0.3, smoothing_slope = 0.01)
    predictions.append(fit.forecast(30))
predictions = np.array(predictions).reshape((-1, 30))
error_holt = np.linalg.norm(predictions - val_dataset.values[:len(predictions)])/len(predictions[0])

# Exponential Smoothing
predictions = []
for row in tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
    fit = ExponentialSmoothing(row, seasonal_periods=3).fit()
    predictions.append(fit.forecast(30))
predictions = np.array(predictions).reshape((-1, 30))
error_exponential = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])

# Arima
predictions = []
for row in tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
    fit = sm.tsa.statespace.SARIMAX(row, seasonal_order=(0, 1, 1, 7)).fit()
    predictions.append(fit.forecast(30))
predictions = np.array(predictions).reshape((-1, 30))
error_arima = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])

# Submitting the prediction file
days = range(1914, 1941 + 1)
time_series_columns = [f'd_{i}' for i in days]
time_series_data = sales_test[time_series_columns]

forecast = pd.DataFrame(time_series_data.iloc[:, -28:].mean(axis=1))
forecast = pd.concat([forecast] * 28, axis=1)
forecast.columns = [f'F{i}' for i in range(1, forecast.shape[1] + 1)]

validation_ids = sales_test['id'].values

predictions = pd.DataFrame(validation_ids, columns=['id'])
forecast = pd.concat([forecast]).reset_index(drop=True)
predictions = pd.concat([predictions, forecast], axis=1)
predictions.to_csv('submission.csv', index=False)