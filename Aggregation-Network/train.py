import os
import argparse
import yaml
import logging
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
from utils.logging import open_log
from utils.tools import load_checkpoint
from utils.visualizer import Visualizer
from models import AttentionNet


def arg_parse():
    parser = argparse.ArgumentParser(description='ClsNet')
    parser.add_argument('-cfg', '--config', default='configs/train/se50-semi-valid0.yaml',
                        type=str, help='load the config file')
    parser.add_argument('--use_html', default=True,
                        type=bool, help='Use html')
    parser.add_argument('--stage', default='train',
                        type=str, help='Which stage: train | valid | infer')
    args = parser.parse_args()
    return args


def seed_reproducer(seed=2333):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def main():
    args = arg_parse()
    config = yaml.load(open(args.config))

    gpus = ','.join([str(i) for i in config['GPUs']])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    seed_reproducer(config['seed'])

    # open log file
    open_log(args, 'train')
    logging.info(args)
    logging.info(config)
    visualizer = Visualizer('Mean-Teacher', config, args)

    logging.info(config['Data_CLASSES'])
    logging.info('Using the network: {}'.format(config['arch']))

    # Student net
    logging.info('Building Student Model')
    StudentModel = AttentionNet.build_model(config, ema=False)
    # Teacher net
    logging.info('Building Teacher Model')
    TeacherModel = AttentionNet.build_model(config, ema=True)

    if config['Using_pretrained_weights']:
        StudentModel.load_pretrained_weights(load_fc=False)

    if config['Cls']['resume'] != None:
        load_checkpoint(StudentModel, config['Cls']['resume'])


    StudentModel.cuda()
    TeacherModel.cuda()

    from utils import net_utils
    optimizer, train_loader, valid_loader = net_utils.prepare_net(config, StudentModel)

    StudentModel = torch.nn.DataParallel(StudentModel)
    TeacherModel = torch.nn.DataParallel(TeacherModel)
    model = [StudentModel, TeacherModel]

    net_utils.train_net(visualizer, optimizer, train_loader, valid_loader, model, config)


if __name__ == '__main__':
    main()
