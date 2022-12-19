import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# train_path =  '../input/cleanedm5data/cleaned_train_data (1).csv'
# test_path = '../input/cleanedm5data/cleaned_test_data (1).csv'
# sample_path = '../input/afcs2021/sample_submission_afcs2021.csv'

calendar = pd.read_csv('../input/afcs2021/calendar_afcs2021.csv')
selling_prices = pd.read_csv('../input/afcs2021/sell_prices_afcs2021.csv')
sample_submission = pd.read_csv('../input/afcs2021/sample_submission_afcs2021.csv')
sales_train_val = pd.read_csv('../input/afcs2021/sales_train_validation_afcs2021.csv')
sales_test = pd.read_csv('../input/afcs2021/sales_test_validation_afcs2021.csv')


ids = sorted(list(set(sales_train_val['id'])))
d_cols = [c for c in sales_train_val.columns if 'd_' in c]
x_1 = sales_train_val.set_index('id')[d_cols].values[0]

fig = make_subplots(rows=1, cols=1)

fig.add_trace(go.Scatter(x=np.arange(len(x_1)), y=x_1, showlegend=False,
                    mode='lines', name="First sample",
                         marker=dict(color="mediumseagreen")),
             row=1, col=1)

fig.update_layout(title_text="Sales")
fig.show()

ids = sorted(list(set(sales_train_val['id'])))
d_cols = [c for c in sales_train_val.columns if 'd_' in c]
x_1 = sales_train_val.set_index('id')[d_cols].values[0]

fig = make_subplots(rows=1, cols=1)

fig.add_trace(go.Scatter(x=np.arange(len(x_1)), y=x_1, showlegend=False,
                    mode='lines', name="First sample",
                         marker=dict(color="mediumseagreen")),
             row=1, col=1)

fig.update_layout(title_text="Sales")
fig.show()

sales = sales_train_val.set_index('id')[d_cols] \
    .T \
    .merge(calendar.set_index('d')['date'],
           left_index=True,
           right_index=True,
           validate='1:1') \
    .set_index('date')

store_list = selling_prices['store_id'].unique()
means = []
fig = go.Figure()
for s in store_list:
    store_items = [c for c in sales.columns if s in c]
    data = sales[store_items].sum(axis=1).rolling(90).mean()
    means.append(np.mean(sales[store_items].sum(axis=1)))
    fig.add_trace(go.Scatter(x=np.arange(len(data)), y=data, name=s))

fig.update_layout(yaxis_title="Sales", xaxis_title="Time", title="Rolling Average Sales vs. Time TX_3 Store")

fig = go.Figure()

for i, s in enumerate(store_list):
    store_items = [c for c in sales.columns if s in c]
    data = sales[store_items].sum(axis=1).rolling(90).mean()
    fig.add_trace(go.Box(x=[s] * len(data), y=data, name=s))

fig.update_layout(yaxis_title="Sales", xaxis_title="Time", title="Rolling Average Sales vs. TX store ")