import logging
from logging import getLogger

import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec, SINE, LightSANs
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import pandas as pd

#%%

parameter_dict = {
    'data_path': 'dataset',
    'USER_ID_FIELD': 'customer_id',
    'ITEM_ID_FIELD': 'article_id',
    'TIME_FIELD': 't_dat',
    'user_inter_num_interval': "[0,inf)",
    'item_inter_num_interval': "[0,inf)",
    'load_col': {'inter': ['customer_id', 'article_id', 't_dat']},
    'neg_sampling': None,
    'epochs': 50,
    'eval_step':2,
    'eval_args': {
        'split': {'RS': [9.9, 0, 0.1]},
        'group_by': 'user',
        'order': 'TO',
        'mode': 'full'}
}

config = Config(model='LightSANs', dataset='hm_data_small', config_dict=parameter_dict)

# init random seed
init_seed(config['seed'], config['reproducibility'])

# logger initialization
init_logger(config)
logger = getLogger()
# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)

# write config info into log
logger.info(config)

#%%

dataset = create_dataset(config)
logger.info(dataset)

#%%

# dataset splitting
train_data, valid_data, test_data = data_preparation(config, dataset)

#%%

# model loading and initialization
model = LightSANs(config, train_data.dataset).to(config['device'])
logger.info(model)

# trainer loading and initialization
trainer = Trainer(config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(train_data)

#%% md


# 4. Create recommendation result from trained model

#%%


from recbole.utils.case_study import full_sort_topk
external_user_ids = dataset.id2token(
    dataset.uid_field, list(range(dataset.user_num)))[1:]#fist element in array is 'PAD'(default of Recbole) ->remove it

#%%

topk_items = []
for internal_user_id in list(range(dataset.user_num))[1:]:
    _, topk_iid_list = full_sort_topk([internal_user_id], model, test_data, k=12, device=config['device'])
    last_topk_iid_list = topk_iid_list[-1]
    external_item_list = dataset.id2token(dataset.iid_field, last_topk_iid_list.cpu()).tolist()
    topk_items.append(external_item_list)
print(len(topk_items))

#%%

external_item_str = [' '.join(x) for x in topk_items]
result = pd.DataFrame(external_user_ids, columns=['customer_id'])
result['prediction'] = external_item_str
print(result.head())


# transfer the index to customer original id

customer_df = pd.read_csv('customers_origin.csv')
customer_id = dict(enumerate(customer_df['customer_id'].tolist()))
result['customer_id'] = result['customer_id'].apply(lambda x: customer_id[int(x)])

#%% md

# 5. Combine result from most bought items and GRU model

#%%

submit_df = pd.read_csv('submission_origin.csv')
submit_df.shape

#%%

submit_df.head()

# %%

submit_df = pd.merge(submit_df, result, on='customer_id', how='outer')
submit_df.head()

#%%

submit_df = submit_df.fillna(-1)
submit_df['prediction'] = submit_df.apply(
    lambda x: x['prediction_y'] if x['prediction_y'] != -1 else x['prediction_x'], axis=1)
submit_df.head()

# %%

submit_df[submit_df['prediction_y'] != -1]


#%%

submit_df = submit_df.drop(columns=['prediction_y', 'prediction_x'])
submit_df.head()

#%%

submit_df.to_csv('submission.csv', index=False)

#%%


