import argparse
import json
import sys
from logging import getLogger
from recbole.utils import init_seed
from hpt_logger import init_logger
# from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
import torch
from recbole.data.transform import construct_transform

from recbole.config import Config

from recbole.data import create_dataset, data_preparation
from recbole.utils import (
    # init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
from dataset.granular_ball import GBList, GranularBall  # 修改为你的定义文件名

from model.hpt import HPT

_original_torch_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = safe_load
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
    config['log_root'] = f'log/{args.dataset}/{args.exp_type}'  # 你想要保存日志的目录
    init_seed(config['seed'], config['reproducibility'])
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
   
    train_data, valid_data, test_data = data_preparation(config, dataset)
    Model=model_dict[args.model]
    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = Model(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    transform = construct_transform(config)

    # trainer loading and initialization
    trainer = Trainer(config, model)
    

    # model training
    best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, show_progress=config["show_progress"]
    )

    
    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )
    
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")