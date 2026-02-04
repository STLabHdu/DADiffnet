import os
import numpy as np

import sys
import yaml
import argparse

sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch

torch.set_num_threads(3)

from src.models.DADiffNet.dadiffnet import DADiffNet
from src.base.engine import BaseEngine
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--if_augment', type=int, default=1)
    parser.add_argument('--if_diff', type=int, default=1)

    parser.add_argument('--tree_emb_len_list', type=list, default=[5])

    parser.add_argument('--graph_num_step', type=int, default=2)
    parser.add_argument('--graph_num_layer', type=int, default=3)
    parser.add_argument('--graph_dim', type=int, default=64)
    parser.add_argument('--graph_out_dim', type=int, default=16)
    parser.add_argument('--graph_dropout', type=float, default=0.2)
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=1)
    parser.add_argument('--if_T_i_D', type=int, default=1)
    parser.add_argument('--if_D_i_W', type=int, default=1)
    parser.add_argument('--temp_dim_tid', type=int, default=32)
    parser.add_argument('--temp_dim_diw', type=int, default=32)
    parser.add_argument('--if_feedback', type=int, default=1)
    parser.add_argument('--time_of_day_size', type=int, default=96)
    parser.add_argument('--day_of_week_size', type=int, default=7)

    parser.add_argument('--adj_type', type=str, default='doubletransition')
    parser.add_argument('--lrate', type=float, default=2e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--clip_grad_value', type=float, default=5)

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='SD')
    # if need to use the data from multiple years, please use underline to separate them, e.g., 2018_2019
    parser.add_argument('--years', type=str, default='2019')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=2023)

    parser.add_argument('--bs', type=int, default=2)
    # seq_len denotes input history length, horizon denotes output future length
    parser.add_argument('--input_len', type=int, default=12)
    parser.add_argument('--output_len', type=int, default=12)
    parser.add_argument('--input_hidden', type=int, default=3)
    parser.add_argument('--output_hidden', type=int, default=1)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=30)

    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, args.adj_type)
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = DADiffNet(node_num=node_num,
                      input_hidden=args.input_hidden,
                      output_hidden=args.output_hidden,
                      model_args=vars(args),
                      supports=supports)

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    steps = [1, 20, 40, 60, 80]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.5)

    engine = BaseEngine(device=device,
                        model=model,
                        dataloader=dataloader,
                        scaler=scaler,
                        sampler=None,
                        loss_fn=loss_fn,
                        lrate=args.lrate,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        clip_grad_value=args.clip_grad_value,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        log_dir=log_dir,
                        logger=logger,
                        seed=args.seed,
                        )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
