import time
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from .tools import *
from .config import *
from .ramps import *
from modules.losses.mse_loss import cls_mse_loss
from data.custom_dataset import *

consistency_criterion_cls = cls_mse_loss


def get_current_consistency_cls_weight(epoch, config):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return config['consistency_cls'] * sigmoid_rampup(epoch, config['consistency_rampup'], type='cls')


def get_current_consistency_att_weight(epoch, config):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if epoch < 20:
        return 0.0
    else:
        return config['consistency_att'] * sigmoid_rampup(epoch, config['consistency_rampup'], type='att')


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def prepare_net(config, model, _use='train'):
    # img_size = (config['img_size'], config['img_size'])
    if _use == 'train':
        if config['optim'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), config['lr'], weight_decay=config['weight_decay'])

        if config['optim'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), config['lr'], weight_decay=config['weight_decay'])

        elif config['optim'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), config['lr'], momentum=config['momentum'],
                                        weight_decay=config['weight_decay'], nesterov=config['nesterov'])

        folds = [fold for fold in range(config['n_fold']) if fold != config['valid_fold']]
        train_dataset = CustomDataset('train', config['FeaRoot'], config['ProRoot'], config['QueryIndex'],
                                      config['TrainFold'], folds)

        labeled_df = pd.read_csv(config['TrainFold'])
        labeled_fold = [i for i in [folds[0]]]
        labeled_df = labeled_df[labeled_df['Fold'].isin(labeled_fold)]
        labeled_fold_name = labeled_df['SegID'].tolist()
        labeled_idxs, unlabeled_idxs = relabel_dataset(train_dataset, labeled_fold_name)
        batch_sampler = TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, config['batchsize'], config['label_bs'])

        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=config['num_workers'], batch_sampler=batch_sampler)

        valid_dataset = CustomDataset('valid', config['FeaRoot'], config['ProRoot'], config['QueryIndex'],
                                      config['TrainFold'], [config['valid_fold']])
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config['batchsize'], shuffle=False,
                                                   num_workers=config['num_workers'], drop_last=False)

        return optimizer, train_loader, valid_loader


def train_net(visualizer, optimizer, train_loader, val_loader, model, config):
    best_auc = 0
    best_f1 = 0
    best_acc = 0
    best_index = [best_acc, best_auc, best_f1]

    cls_criterion = nn.NLLLoss()

    if config['lr_decay'] == None:
        lr_decay = 0.1
    else:  
        lr_decay = config['lr_decay']

    for epoch in range(1, config['num_epoch']+1):
        adjust_learning_rate(optimizer, epoch - 1, config['num_epoch'], config['lr'], config['lr_decay_freq'], lr_decay)

        train(visualizer, train_loader, model, optimizer, epoch, config, cls_criterion)

        if epoch % config['valid_freq'] == 0:
            best_index = valid_net(val_loader, model, config, best_index, epoch)
            logging.info('Valid-Cls: Best ACC  update to: {:.4f}'.format(best_index[0]))
            logging.info('Valid-Cls: Best AUC  update to: {:.4f}'.format(best_index[1]))
            logging.info('Valid-Cls: Best F1   update to: {:.4f}'.format(best_index[2]))


def valid_net(val_loader, model, config, best_index, epoch):
    result_s, result_t = valid(val_loader, model, config)
    StudentModel, TeacherModel = model
    m_auc_s, all_auc_s, m_acc_s, all_acc_s, m_f1_s, all_f1_s = result_s
    m_auc_t, all_auc_t, m_acc_t, all_acc_t, m_f1_t, all_f1_t = result_t
    best_acc, best_auc, best_f1 = best_index

    logging.info('Valid-Cls: Student Model')
    logging.info('Valid-Cls: Mean ACC: {:.4f}, Mean AUC: {:.4f}, Mean F1: {:.4f}'.format(m_acc_s, m_auc_s, m_f1_s))
    print_result('Valid-Cls: ACC for All Classes: ', all_acc_s, config['Data_CLASSES'])
    print_result('Valid-Cls: AUC for All Classes: ', all_auc_s, config['Data_CLASSES'])
    print_result('Valid-Cls: F1  for All Classes: ', all_f1_s,  config['Data_CLASSES'])
    logging.info('Valid-Cls: Teacher Model')
    logging.info('Valid-Cls: Mean ACC: {:.4f}, Mean AUC: {:.4f}, Mean F1: {:.4f}'.format(m_acc_t, m_auc_t, m_f1_t))
    print_result('Valid-Cls: ACC for All Classes: ', all_acc_t, config['Data_CLASSES'])
    print_result('Valid-Cls: AUC for All Classes: ', all_auc_t, config['Data_CLASSES'])
    print_result('Valid-Cls: F1  for All Classes: ', all_f1_t,  config['Data_CLASSES'])

    m_acc = max(m_acc_s, m_acc_t)
    m_auc = max(m_auc_s, m_auc_t)
    m_f1  = max(m_f1_s, m_f1_t)

    if m_acc >= best_acc:
        save_checkpoint(StudentModel, config['arch'] + '_S_valid' + str(config['valid_fold']), epoch, _best='acc', best=m_acc_s)
        save_checkpoint(TeacherModel, config['arch'] + '_T_valid' + str(config['valid_fold']), epoch, _best='acc', best=m_acc_t)
        best_acc = m_acc
    if m_auc >= best_auc:
        save_checkpoint(StudentModel, config['arch'] + '_S_valid' + str(config['valid_fold']), epoch, _best='auc', best=m_auc_s)
        save_checkpoint(TeacherModel, config['arch'] + '_T_valid' + str(config['valid_fold']), epoch, _best='auc', best=m_auc_t)
        best_auc = m_auc
    if m_f1  >= best_f1:
        save_checkpoint(StudentModel, config['arch'] + '_S_valid' + str(config['valid_fold']), epoch, _best='f1', best=m_f1_s)
        save_checkpoint(TeacherModel, config['arch'] + '_T_valid' + str(config['valid_fold']), epoch, _best='f1', best=m_f1_t)
        best_f1 = m_f1

    return [best_acc, best_auc, best_f1]


def train(visualizer, train_loader, model, optimizer, epoch, config, cls_criterion):
    StudentModel, TeacherModel = model
    losses = AverageMeter()
    cls_losses = AverageMeter()
    consiscls_losses = AverageMeter()
    batch_time = AverageMeter()
    cls_ACCs = AverageMeter()
    cls_AUCs = AverageMeter()
    cls_F1s = AverageMeter()
    num_classes = len(config['Data_CLASSES'])

    StudentModel.train()
    TeacherModel.train()
    end = time.time()

    for i, (input, ema_input, label, name) in enumerate(train_loader):

        with torch.autograd.set_detect_anomaly(True):
            fea1, pro1 = input
            fea2, pro2 = ema_input
            bs   = fea1.size(0)
            label_bs = config['label_bs']

            visualizer.reset()
            errors_ret = OrderedDict()

            fea1 = fea1.cuda()
            pro1 = pro1.cuda()
            fea2 = fea2.cuda()
            pro2 = pro2.cuda()
            label = label.cuda().squeeze()

            output_s = StudentModel((fea1, pro1))
            output_t = TeacherModel((fea2, pro2))

            ### Classification
            probe = torch.softmax(output_s, dim=1).clamp(min=1e-6)
            cls_loss = cls_criterion(torch.log(probe[:label_bs]), label[:label_bs])

            ### Classification Consistency
            consistency_weight_att = get_current_consistency_att_weight(epoch, config)
            consistency_loss_cls = consistency_weight_att * consistency_criterion_cls(output_s,
                                                                                      output_t)

            ## Ours
            if epoch < 20:
                total_loss = loss_cls * cls_loss
            else:
                total_loss = loss_cls * cls_loss + consistency_loss_cls

            errors_ret['Loss'] = float(total_loss)
            errors_ret['ClsLoss'] = float(cls_loss)
            errors_ret['ConsisClsLoss'] = float(consistency_loss_cls)

            losses.update(total_loss.item(), bs)
            cls_losses.update(cls_loss.item(), bs)
            consiscls_losses.update(consistency_loss_cls.item(), bs)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            update_ema_variables(StudentModel, TeacherModel, config['ema_decay'], epoch)

            m_acc, _ = recall(probe.cpu().detach().numpy(), label.cpu().detach().numpy(), config)
            cls_ACCs.update(m_acc, bs)
            m_auc, _ = calculate_auc(probe.cpu().detach().numpy(), label.cpu().detach().numpy(), config)
            cls_AUCs.update(m_auc, bs)
            m_f1, _ = calculate_f1(probe.cpu().detach().numpy(), label.cpu().detach().numpy(), config)
            cls_F1s.update(m_f1, bs)

            batch_time.update(time.time() - end)
            end = time.time()
            if i % config['print_freq'] == 0:
                logging.info('Epoch: [{}][{}/{}]\t'
                             'ConsistencyWeightAtt: {:.4f} '
                             'Loss: {loss.val:.4f} ({loss.avg:.4f}) '
                             'ClsLoss: {cls_loss.val:.4f} ({cls_loss.avg:.4f}) '
                             'ConsisClsLoss: {concls_loss.val:.4f} ({concls_loss.avg:.4f}) '
                             'ClsF1: {cls_f1.val:.4f} ({cls_f1.avg:.4f}) '.format(
                    epoch, i, len(train_loader), consistency_weight_att, loss=losses, cls_loss=cls_losses,
                    concls_loss=consiscls_losses, cls_f1=cls_F1s))


def valid(valid_loader, model, config):
    StudentModel, TeacherModel = model
    batch_time = AverageMeter()
    StudentModel.eval()
    TeacherModel.eval()

    num_classes = len(config['Data_CLASSES'])

    with torch.no_grad():
        end = time.time()
        for i, (input, ema_input, label, name) in enumerate(valid_loader):
            fea1, pro1 = input
            fea2, pro2 = ema_input

            bs = fea1.size(0)

            fea1 = fea1.cuda()
            pro1 = pro1.cuda()
            fea2 = fea2.cuda()
            pro2 = pro2.cuda()
            label = label.cuda()

            output_s = StudentModel((fea1, pro1))
            output_t = TeacherModel((fea2, pro2))

            probe_s = torch.softmax(output_s, dim=1)
            probe_t = torch.softmax(output_t, dim=1)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % (config['print_freq'] * config['batchsize']) == 0:
                logging.info('Valid: [{}/{}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(
                       i, len(valid_loader), batch_time=batch_time))

            if i == 0:
                y_true = label.cpu().detach().numpy()
                y_pred_s = probe_s.cpu().detach().numpy()
                y_pred_t = probe_t.cpu().detach().numpy()
            else:
                y_true = np.concatenate((y_true, label.cpu().detach().numpy()), axis=0)
                y_pred_s = np.concatenate((y_pred_s, probe_s.cpu().detach().numpy()), axis=0)
                y_pred_t = np.concatenate((y_pred_t, probe_t.cpu().detach().numpy()), axis=0)

        m_auc_s, all_auc_s = calculate_auc(y_pred_s, y_true, config)
        m_acc_s, all_acc_s = recall(y_pred_s, y_true, config)
        m_f1_s, all_f1_s = calculate_f1(y_pred_s, y_true, config)

        m_auc_t, all_auc_t = calculate_auc(y_pred_t, y_true, config)
        m_acc_t, all_acc_t = recall(y_pred_t, y_true, config)
        m_f1_t, all_f1_t = calculate_f1(y_pred_t, y_true, config)

        return [m_auc_s, all_auc_s, m_acc_s, all_acc_s, m_f1_s, all_f1_s], \
               [m_auc_t, all_auc_t, m_acc_t, all_acc_t, m_f1_t, all_f1_t]

