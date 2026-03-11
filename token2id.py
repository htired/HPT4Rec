import argparse
import json
import os
import pickle
import sys
import logging
from logging import getLogger
import numpy as np
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
from model.granular_ball import GBList, GranularBall  # 修改为你的定义文件名

from model.hpt import HPT

if __name__ == '__main__':

    # === 添加命令行参数解析 ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='HPT')
    parser.add_argument('--dataset', type=str, default='amazon-baby')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--exp_type', type=str, default="overall")
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--zeta', type=float, default=3)
    parser.add_argument('--rho', type=float, default=0.3)
    parser.add_argument('--p', type=int, default=50)
    parser.add_argument('--L', type=int, default=3)
    args = parser.parse_args()

    model_dict = {
        'HPT': HPT,
    }
    config_dict = {
        'HPT': "model/config.yaml",
    }
    if args.model not in model_dict:
        raise ValueError(f"Unsupported model: {args.model}")
    config = Config(model=model_dict[args.model], config_file_list=[config_dict[args.model]])
    config['data_path'] = "dataset/"+args.dataset
    config['dataset'] = args.dataset
    config['seed'] = args.seed
    config['alpha'] = args.alpha
    config['beta'] = args.beta
    config['zeta'] = args.zeta
    config['rho'] = args.rho
    config['p'] = args.p
    config['L'] = args.L
    init_seed(config['seed'], config['reproducibility'])
    # dataset filtering
    dataset = create_dataset(config)

    os.makedirs(os.path.dirname(f'dataset/{args.dataset}/backbone/'), exist_ok=True)
    with open(f'dataset/{args.dataset}/backbone/user_token_to_id_all.json', 'w') as f:
        json.dump(dataset.field2token_id['user_id'], f, indent=2)
    with open(f'dataset/{args.dataset}/backbone/item_token_to_id_all.json', 'w') as f:
        json.dump(dataset.field2token_id['item_id'], f, indent=2)
    print("Saved user_token_to_id_all.json and item_token_to_id_all.json")
    