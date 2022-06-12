from recbole.quick_start import run_recbole
from logging import getLogger
from recbole.model.sequential_recommender import GRU4Rec, SINE, LightSANs
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color
import pandas as pd

parameter_file = "./recbole/properties/dataset/sample.yaml"
# run_recbole(model='LightSANs ', dataset='hm_data', config_file_list=[parameter_file])
model='LightSANs'
dataset='hm_data'
saved = True
config_file_list=[parameter_file]
config = Config(model=model, dataset=dataset, config_file_list=config_file_list)
init_seed(config['seed'], config['reproducibility'])
# logger initialization
init_logger(config)
logger = getLogger()

logger.info(config)

# dataset filtering
dataset = create_dataset(config)
logger.info(dataset)

# dataset splitting
train_data, valid_data, test_data = data_preparation(config, dataset)

# model loading and initialization
init_seed(config['seed'], config['reproducibility'])
model = LightSANs(config, train_data.dataset).to(config['device'])
logger.info(model)

# trainer loading and initialization
trainer = Trainer(config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(
    train_data, valid_data, saved=saved, show_progress=config['show_progress']
)

# model evaluation
test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
logger.info(set_color('test result', 'yellow') + f': {test_result}')

print({
    'best_valid_score': best_valid_score,
    'valid_score_bigger': config['valid_metric_bigger'],
    'best_valid_result': best_valid_result,
    'test_result': test_result
})






# GRU4Rec
# 02 Jun 04:48    INFO  best valid : OrderedDict([('recall@10', 0.0281), ('mrr@10', 0.0137), ('ndcg@10', 0.0172), ('hit@10', 0.0281), ('precision@10', 0.0028)])
# 02 Jun 04:48    INFO  test result: OrderedDict([('recall@10', 0.0262), ('mrr@10', 0.0122), ('ndcg@10', 0.0155), ('hit@10', 0.0262), ('precision@10', 0.0026)])

# Score: 0.02291
# Public score: 0.02263

# SINE
# 02 Jun 04:54    INFO  best valid : OrderedDict([('recall@10', 0.092), ('mrr@10', 0.0609), ('ndcg@10', 0.0685), ('hit@10', 0.092), ('precision@10', 0.0092)])
# 02 Jun 04:54    INFO  test result: OrderedDict([('recall@10', 0.0792), ('mrr@10', 0.0521), ('ndcg@10', 0.0586), ('hit@10', 0.0792), ('precision@10', 0.0079)])

# LightSANs
# 02 Jun 04:58    INFO  best valid : OrderedDict([('recall@10', 0.1093), ('mrr@10', 0.0977), ('ndcg@10', 0.1005), ('hit@10', 0.1093), ('precision@10', 0.0109)])
# 02 Jun 04:58    INFO  test result: OrderedDict([('recall@10', 0.1704), ('mrr@10', 0.1556), ('ndcg@10', 0.1591), ('hit@10', 0.1704), ('precision@10', 0.017)])