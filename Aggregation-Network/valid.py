import os
import argparse
import yaml
import logging
import numpy as np
import pandas as pd
import random
from data.custom_dataset import CustomDataset
from utils.logging import open_log
from utils.tools import *
from models import AttentionNet
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from imblearn.metrics import sensitivity_score, specificity_score


def arg_parse():
    parser = argparse.ArgumentParser(
        description='ClsNet')
    parser.add_argument('-cfg', '--config', default='configs/valid/se50-subject-valid0.yaml',
                        type=str, help='load the config file')
    parser.add_argument('--stage', default='valid',
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


def get_one_hot(data, num_classes):
    data = data.reshape(-1)
    data = np.eye(num_classes)[data]
    return data


def main():
    args = arg_parse()
    config = yaml.load(open(args.config))

    gpus = ','.join([str(i) for i in config['GPUs']])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    seed_reproducer(config['seed'])
    
    # open log file
    open_log(args, args.stage)
    logging.info(args)
    logging.info(config)
    config['model_name'] = args.config.split('/')[-1].split('.')[0]

    logging.info(config['Data_CLASSES'])
    logging.info('Using the network: {}'.format(config['arch']))

    # set net
    logging.info('Building ClsModel')
    AttentionModel = AttentionNet.build_model(config, ema=False)

    assert config['TestModel'] != ''  # must load a trained model
    logging.info('Resuming network: {}'.format(config['TestModel']))
    load_checkpoint(AttentionModel, config['TestModel'])

    AttentionModel.cuda()

    valid_dataset = CustomDataset('valid', config['FeaRoot'], config['ProRoot'], config['QueryIndex'],
                                  config['TrainFold'], [config['valid_fold']])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config['batchsize'], shuffle=False,
                                               num_workers=config['num_workers'], drop_last=False)

    AttentionModel = torch.nn.DataParallel(AttentionModel)
    AttentionModel.eval()

    num_classes = len(config['Data_CLASSES'])
    AUROCs, Accus, Senss, Specs, F1 = [], [], [], [], []

    with torch.no_grad():
        for i, (input, ema_input, label, name) in enumerate(valid_loader):
            fea1, pro1 = input
            bs = fea1.size(0)

            fea1 = fea1.cuda()
            pro1 = pro1.cuda()
            label = label.cuda()

            output = AttentionModel((fea1, pro1))
            probe = torch.softmax(output, dim=1)

            if i == 0:
                y_true = label.cpu().detach().numpy()
                y_pred = probe.cpu().detach().numpy()
            else:
                y_true = np.concatenate((y_true, label.cpu().detach().numpy()), axis=0)
                y_pred = np.concatenate((y_pred, probe.cpu().detach().numpy()), axis=0)

        y_true_one_hot = get_one_hot(y_true, num_classes)
        y_pred_one_hot = np.argmax(y_pred, axis=1)
        y_pred_one_hot = get_one_hot(y_pred_one_hot, num_classes)

        for i in range(num_classes):
            try:
                AUROCs.append(roc_auc_score(y_true_one_hot[:, i], y_pred[:, i]))
            except ValueError as error:
                print('Error in computing AUC for {}.\n Error msg:{}'.format(i, error))
                AUROCs.append(0)

            try:
                Accus.append(accuracy_score(y_true_one_hot[:, i], y_pred_one_hot[:, i]))
            except ValueError as error:
                print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
                Accus.append(0)

            try:
                Senss.append(sensitivity_score(y_true_one_hot[:, i], y_pred_one_hot[:, i]))
            except ValueError:
                print('Error in computing sensitivity for {}.'.format(i))
                Senss.append(0)

            try:
                Specs.append(specificity_score(y_true_one_hot[:, i], y_pred_one_hot[:, i]))
            except ValueError:
                print('Error in computing specificity for {}.'.format(i))
                Specs.append(0)

            try:
                F1.append(f1_score(y_true_one_hot[:, i], y_pred_one_hot[:, i]))
            except ValueError:
                print('Error in computing F1-score for {}.'.format(i))
                F1.append(0)

        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        F1_avg = np.array(F1).mean()

        logging.info('Valid-Cls: Mean AUC: {:.4f}'.format(AUROC_avg))
        logging.info('Valid-Cls: Mean ACC: {:.4f}'.format(Accus_avg))
        logging.info('Valid-Cls: Mean SEN: {:.4f}'.format(Senss_avg))
        logging.info('Valid-Cls: Mean SPE: {:.4f}'.format(Specs_avg))
        logging.info('Valid-Cls: Mean F1 : {:.4f}'.format(F1_avg))
        print_result('Valid-Cls: AUC for All Classes: ', AUROCs, config['Data_CLASSES'])
        print_result('Valid-Cls: ACC for All Classes: ', Accus, config['Data_CLASSES'])
        print_result('Valid-Cls: SEN for All Classes: ', Senss, config['Data_CLASSES'])
        print_result('Valid-Cls: SPE for All Classes: ', Specs, config['Data_CLASSES'])
        print_result('Valid-Cls: F1  for All Classes: ', F1, config['Data_CLASSES'])


if __name__ == '__main__':
    main()
