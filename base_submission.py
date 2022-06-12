import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import gc
import pandas as cudf

N = 12
df  = cudf.read_csv('transactions_train.csv',
                            usecols= ['t_dat', 'customer_id', 'article_id'],
                            dtype={'article_id': 'int32', 't_dat': 'string', 'customer_id': 'string'})
df ['customer_id'] = df ['customer_id'].str[-16:].apply(int, base=16).astype('int64')

df['t_dat'] = cudf.to_datetime(df['t_dat'])
last_ts = df['t_dat'].max()

tmp = df[['t_dat']].copy()
tmp['dow'] = tmp['t_dat'].dt.dayofweek
tmp['ldbw'] = tmp['t_dat'] - pd.TimedeltaIndex(tmp['dow'] - 1, unit='D')
tmp.loc[tmp['dow'] >=2 , 'ldbw'] = tmp.loc[tmp['dow'] >=2 , 'ldbw'] + pd.TimedeltaIndex(np.ones(len(tmp.loc[tmp['dow'] >=2])) * 7, unit='D')

df['ldbw'] = tmp['ldbw'].values

weekly_sales = df.drop('customer_id', axis=1).groupby(['ldbw', 'article_id']).count().reset_index()
weekly_sales = weekly_sales.rename(columns={'t_dat': 'count'})



df = df.merge(weekly_sales, on=['ldbw', 'article_id'], how = 'left')

weekly_sales = weekly_sales.reset_index().set_index('article_id')

df = df.merge(
    weekly_sales.loc[weekly_sales['ldbw']==last_ts, ['count']],
    on='article_id', suffixes=("", "_targ"))

df['count_targ'].fillna(0, inplace=True)
del weekly_sales

df['quotient'] = df['count_targ'] / df['count']

target_sales = df.drop('customer_id', axis=1).groupby('article_id')['quotient'].sum()
general_pred = target_sales.nlargest(N).index.tolist()
general_pred = ['0' + str(article_id) for article_id in general_pred]
general_pred_str =  ' '.join(general_pred)
del target_sales


purchase_dict = {}

tmp = df.copy()
tmp['x'] = ((last_ts - tmp['t_dat']) / np.timedelta64(1, 'D')).astype(int)
tmp['dummy_1'] = 1
tmp['x'] = tmp[["x", "dummy_1"]].max(axis=1)

a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
tmp['y'] = a / np.sqrt(tmp['x']) + b * np.exp(-c*tmp['x']) - d

tmp['dummy_0'] = 0
tmp['y'] = tmp[["y", "dummy_0"]].max(axis=1)
tmp['value'] = tmp['quotient'] * tmp['y']

tmp = tmp.groupby(['customer_id', 'article_id']).agg({'value': 'sum'})
tmp = tmp.reset_index()

tmp = tmp.loc[tmp['value'] > 100]
tmp['rank'] = tmp.groupby("customer_id")["value"].rank("dense", ascending=False)
tmp = tmp.loc[tmp['rank'] <= 12]

# for customer_id in tmp['customer_id'].unique():
#     purchase_dict[customer_id] = {}

# for customer_id, article_id, value in zip(tmp['customer_id'], tmp['article_id'], tmp['value']):
#     purchase_dict[customer_id][article_id] = value

purchase_df = tmp.sort_values(['customer_id', 'value'], ascending = False).reset_index(drop = True)
purchase_df['prediction'] = '0' + purchase_df['article_id'].astype(str) + ' '
purchase_df = purchase_df.groupby('customer_id').agg({'prediction': sum}).reset_index()
purchase_df['prediction'] = purchase_df['prediction'].str.strip()
purchase_df = cudf.DataFrame(purchase_df)

sub  = cudf.read_csv('sample_submission.csv',
                            usecols= ['customer_id'],
                            dtype={'customer_id': 'string'})

sub['customer_id2'] = sub['customer_id'].str[-16:].apply(int, base=16).astype('int64')

sub = sub.merge(purchase_df, left_on = 'customer_id2', right_on = 'customer_id', how = 'left',
               suffixes = ('', '_ignored'))

# sub = sub.to_pandas()
sub['prediction'] = sub['prediction'].fillna(general_pred_str)
sub['prediction'] = sub['prediction'] + ' ' +  general_pred_str
sub['prediction'] = sub['prediction'].str.strip()
sub['prediction'] = sub['prediction'].str[:131]
sub = sub[['customer_id', 'prediction']]
sub.to_csv(f'submission_origin.csv',index=False)