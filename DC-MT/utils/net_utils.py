import time
import cv2
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import albumentations
import pandas as pd
from .tools import *
from .config import *
from .ramps import *
from modules.losses.mse_loss import cls_mse_loss, att_mse_loss
from torchvision import transforms
from data.custom_dataset import CustomDataset
from tqdm import tqdm

mask_mse_loss_func = att_mse_loss
consistency_criterion_cls = cls_mse_loss
consistency_criterion_att = att_mse_loss


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
        train_dataset = CustomDataset('train', config['DataRoot'], config['TrainFold'], folds,
                                      transform=albumentations.Compose([
                                          albumentations.Resize(config['img_size'], config['img_size']),
                                          albumentations.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.9),
                                          albumentations.OneOf([
                                              albumentations.Blur(blur_limit=3, p=1),
                                              albumentations.MedianBlur(blur_limit=3, p=1)
                                          ], p=0.5),
                                          albumentations.GaussNoise(var_limit=(0.0, 0.1), mean=0, per_channel=True, always_apply=False, p=1.0),
                                          albumentations.HorizontalFlip(p=0.5),
                                          albumentations.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.01,
                                                                          rotate_limit=3,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          border_mode=cv2.BORDER_CONSTANT, p=0.5),
                                      ]),
                                     )

        labeled_df = pd.read_csv(config['TrainFold'])
        labeled_fold = [i for i in [folds[0]]]
        labeled_df = labeled_df[labeled_df['Fold'].isin(labeled_fold)]
        labeled_fold_name = labeled_df['Path'].tolist()
        labeled_idxs, unlabeled_idxs = relabel_dataset(train_dataset, labeled_fold_name)
        batch_sampler = TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, config['batchsize'], config['label_bs'])

        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=config['num_workers'], batch_sampler=batch_sampler)

        valid_dataset = CustomDataset('valid', config['DataRoot'], config['TrainFold'], [config['valid_fold']],
                                      transform=albumentations.Compose([
                                          albumentations.Resize(config['img_size'], config['img_size']),
                                      ])
                                      )
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config['batchsize'], shuffle=False,
                                                   num_workers=config['num_workers'], drop_last=False)

        return optimizer, train_loader, valid_loader

    elif _use == 'infer_subject':
        infer_dataset = InferDataset('infer', config['DataRoot'], config['TestFold'],
                                     transform=albumentations.Compose([
                                         albumentations.Resize(config['img_size'], config['img_size']),
                                     ])
                                     )
        infer_loader = torch.utils.data.DataLoader(infer_dataset, batch_size=1, shuffle=False,
                                                   num_workers=config['num_workers'], drop_last=False)
        return infer_loader


def train_net(visualizer, optimizer, train_loader, val_loader, model, config):
    best_auc = 0
    best_f1 = 0
    best_acc = 0
    best_tiou = 0.0
    best_tior = 0.0
    best_index = [best_acc, best_auc, best_f1, best_tiou, best_tior]

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
            logging.info('Valid-Cls: Best TIOU update to: {:.4f}'.format(best_index[3]))
            logging.info('Valid-Cls: Best TIOR update to: {:.4f}'.format(best_index[4]))


def valid_net(val_loader, model, config, best_index, epoch):
    result_s, result_t, TIOU, TIOR = valid(val_loader, model, config)
    StudentModel, TeacherModel = model
    m_auc_s, all_auc_s, m_acc_s, all_acc_s, m_f1_s, all_f1_s = result_s
    m_auc_t, all_auc_t, m_acc_t, all_acc_t, m_f1_t, all_f1_t = result_t
    TIOU_s, TIOU_t = TIOU
    TIOR_s, TIOR_t = TIOR
    best_acc, best_auc, best_f1, best_tiou, best_tior = best_index

    mTIOU_s = 0.
    mTIOU_t = 0.
    assert TIOU_s.shape[1] == TIOU_t.shape[1], "TIOU dimension error"
    len_TIOU = TIOU_s.shape[1]
    for idx in range(len(config['Data_CLASSES'])):
        mTIOU_s += TIOU_s[idx].sum() / float(len_TIOU)
        mTIOU_t += TIOU_t[idx].sum() / float(len_TIOU)
    mTIOU_s /= float(len(config['Data_CLASSES']))
    mTIOU_t /= float(len(config['Data_CLASSES']))

    mTIOR_s = 0.
    mTIOR_t = 0.
    assert TIOR_s.shape[1] == TIOR_t.shape[1], "TIOR dimension error"
    len_TIOR = TIOR_s.shape[1]
    for idx in range(len(config['Data_CLASSES'])):
        mTIOR_s += TIOR_s[idx].sum() / float(len_TIOR)
        mTIOR_t += TIOR_t[idx].sum() / float(len_TIOR)
    mTIOR_s /= float(len(config['Data_CLASSES']))
    mTIOR_t /= float(len(config['Data_CLASSES']))

    logging.info('Valid-Cls: Student Model')
    logging.info('Valid-Cls: Mean ACC: {:.4f}, Mean AUC: {:.4f}, Mean F1: {:.4f}, Mean TIoU: {:.4f}, Mean TIoR: {:.4f}'.format(m_acc_s, m_auc_s, m_f1_s, mTIOU_s, mTIOR_s))
    print_result('Valid-Cls: ACC for All Classes: ', all_acc_s, config['Data_CLASSES'])
    print_result('Valid-Cls: AUC for All Classes: ', all_auc_s, config['Data_CLASSES'])
    print_result('Valid-Cls: F1  for All Classes: ', all_f1_s,  config['Data_CLASSES'])
    print_thresh_result('Valid-TIoU: ', TIOU_s, thresh_TIOU, config['Data_CLASSES'])
    print_thresh_result('Valid-TIoR: ', TIOR_s, thresh_TIOR, config['Data_CLASSES'])
    logging.info('Valid-Cls: Teacher Model')
    logging.info('Valid-Cls: Mean ACC: {:.4f}, Mean AUC: {:.4f}, Mean F1: {:.4f}, Mean TIoU: {:.4f}, Mean TIoR: {:.4f}'.format(m_acc_t, m_auc_t, m_f1_t, mTIOU_t, mTIOR_t))
    print_result('Valid-Cls: ACC for All Classes: ', all_acc_t, config['Data_CLASSES'])
    print_result('Valid-Cls: AUC for All Classes: ', all_auc_t, config['Data_CLASSES'])
    print_result('Valid-Cls: F1  for All Classes: ', all_f1_t, config['Data_CLASSES'])
    print_thresh_result('Valid-TIoU: ', TIOU_t, thresh_TIOU, config['Data_CLASSES'])
    print_thresh_result('Valid-TIoR: ', TIOR_t, thresh_TIOR, config['Data_CLASSES'])

    m_acc = max(m_acc_s, m_acc_t)
    m_auc = max(m_auc_s, m_auc_t)
    m_f1 = max(m_f1_s, m_f1_t)
    m_tiou = max(mTIOU_s, mTIOU_t)
    m_tior = max(mTIOR_s, mTIOR_t)

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
    if m_tiou  >= best_tiou:
        save_checkpoint(StudentModel, config['arch'] + '_S_valid' + str(config['valid_fold']), epoch, _best='tiou', best=mTIOU_s)
        save_checkpoint(TeacherModel, config['arch'] + '_T_valid' + str(config['valid_fold']), epoch, _best='tiou', best=mTIOU_t)
        best_tiou = m_tiou
    if m_tior  >= best_tior:
        save_checkpoint(StudentModel, config['arch'] + '_S_valid' + str(config['valid_fold']), epoch, _best='tior', best=mTIOR_s)
        save_checkpoint(TeacherModel, config['arch'] + '_T_valid' + str(config['valid_fold']), epoch, _best='tior', best=mTIOR_t)
        best_tior = m_tior

    return [best_acc, best_auc, best_f1, best_tiou, best_tior]


def train(visualizer, train_loader, model, optimizer, epoch, config, cls_criterion):
    StudentModel, TeacherModel = model
    losses = AverageMeter()
    cls_losses = AverageMeter()
    attmse_losses = AverageMeter()
    attbound_losses = AverageMeter()
    consiscls_losses = AverageMeter()
    consisatt_losses = AverageMeter()
    batch_time = AverageMeter()
    cls_ACCs = AverageMeter()
    cls_AUCs = AverageMeter()
    cls_F1s = AverageMeter()

    StudentModel.train()
    TeacherModel.train()
    end = time.time()

    for i, (input, ema_input, label, _, _) in enumerate(train_loader):

        with torch.autograd.set_detect_anomaly(True):
            image1, masks1 = input
            image2, _ = ema_input

            im_h = image1.size(2)
            im_w = image1.size(3)
            bs   = image1.size(0)
            label_bs = config['label_bs']

            visualizer.reset()
            visual_ret = OrderedDict()
            errors_ret = OrderedDict()

            image1 = image1.cuda()
            masks1 = masks1.cuda()
            image2 = image2.cuda()
            masks1 = masks1.unsqueeze(1)
            label = label.cuda()

            visual_ret['input'] = image1
            masks_vis = visual_masks(masks1, im_h, im_w)
            visual_ret['mask'] = masks_vis

            output_s, cam_refined_s, _ = StudentModel(image1)
            output_t, cam_refined_t, _ = TeacherModel(image2)

            class_idx = label.cpu().long().numpy()
            for index, idx in enumerate(class_idx):
                tmp1 = cam_refined_s[index, idx, :, :].unsqueeze(0).unsqueeze(1)
                tmp2 = cam_refined_t[index, idx, :, :].unsqueeze(0).unsqueeze(1)
                if index == 0:
                    cam_refined_class_s = tmp1
                    cam_refined_class_t = tmp2
                else:
                    cam_refined_class_s = torch.cat((cam_refined_class_s, tmp1), dim=0)
                    cam_refined_class_t = torch.cat((cam_refined_class_t, tmp2), dim=0)
            cam_refined_s = cam_refined_class_s
            cam_refined_t = cam_refined_class_t

            ### Classification
            probe = torch.softmax(output_s, dim=1)
            cls_loss = cls_criterion(torch.log(probe[:label_bs]), label[:label_bs])

            ### Attention
            ## MSE loss
            mask_loss = mask_mse_loss_func(masks1[:label_bs], cam_refined_s[:label_bs])
            ## Bound loss
            bound_loss = torch.tensor(1) - torch.min(masks1[:label_bs], cam_refined_s[:label_bs]).sum((2, 3)) / torch.clamp(cam_refined_s[:label_bs].sum((2, 3)), min=1e-5)
            bound_loss = bound_loss.sum() / bs

            gcams_vis = visual_masks(cam_refined_s.float(), im_h, im_w)
            visual_ret['attention'] = gcams_vis

            ### Attention Consistency
            consistency_weight_att = get_current_consistency_att_weight(epoch, config)
            consistency_loss_att = consistency_weight_att * consistency_criterion_att(cam_refined_s[label_bs:],
                                                                                      cam_refined_t[label_bs:])

            ### Classification Consistency
            consistency_weight_cls = get_current_consistency_cls_weight(epoch, config)
            consistency_loss_cls = consistency_weight_cls * consistency_criterion_cls(output_s,
                                                                                      output_t)

            ## Ours
            if epoch < 20:
                total_loss = loss_cls * cls_loss + loss_masks * mask_loss + loss_bound * bound_loss
            else:
                total_loss = loss_cls * cls_loss + loss_masks * mask_loss + loss_bound * bound_loss + consistency_loss_cls + consistency_loss_att

            errors_ret['ClsLoss'] = float(cls_loss)
            errors_ret['AttMseLoss'] = float(mask_loss)
            errors_ret['AttBoundLoss'] = float(bound_loss)
            errors_ret['ConsisClsLoss'] = float(consistency_loss_cls)
            errors_ret['ConsisAttLoss'] = float(consistency_loss_att)
            errors_ret['Loss'] = float(total_loss)

            losses.update(total_loss.item(), bs)
            cls_losses.update(cls_loss.item(), bs)
            attmse_losses.update(mask_loss.item(), bs)
            attbound_losses.update(bound_loss.item(), bs)
            consiscls_losses.update(consistency_loss_cls.item(), bs)
            consisatt_losses.update(consistency_loss_att.item(), bs)

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
                             'AttMseloss: {attmse_loss.val:.4f} ({attmse_loss.avg:.4f}) '
                             'AttBndLoss: {attbnd_loss.val:.4f} ({attbnd_loss.avg:.4f}) '
                             'ConsisClsLoss: {concls_loss.val:.4f} ({concls_loss.avg:.4f}) '
                             'ConsisAttLoss: {conatt_loss.val:.4f} ({conatt_loss.avg:.4f}) '
                             'ClsF1: {cls_f1.val:.4f} ({cls_f1.avg:.4f}) '.format(
                    epoch, i, len(train_loader), consistency_weight_att, loss=losses, cls_loss=cls_losses, attmse_loss=attmse_losses,
                    attbnd_loss=attbound_losses, concls_loss=consiscls_losses, conatt_loss=consisatt_losses, cls_f1=cls_F1s))

                if config['display_id'] > 0:
                    visualizer.plot_current_losses(epoch, float(i) / float(len(train_loader)), errors_ret)
            if i % config['display_freq'] == 0:
                visualizer.display_current_results(visual_ret, class_idx[0], epoch, save_result=False)


def valid(valid_loader, model, config):
    StudentModel, TeacherModel = model
    batch_time = AverageMeter()
    StudentModel.eval()
    TeacherModel.eval()

    num_classes = len(config['Data_CLASSES'])
    counts = np.zeros(num_classes)
    TIOU_s = np.zeros((num_classes, len(thresh_TIOU)))
    TIOR_s = np.zeros((num_classes, len(thresh_TIOR)))
    TIOU_t = np.zeros((num_classes, len(thresh_TIOU)))
    TIOR_t = np.zeros((num_classes, len(thresh_TIOR)))

    with torch.no_grad():
        end = time.time()
        for i, (input, _, label, _, _) in enumerate(valid_loader):
            image, masks = input

            image = image.cuda()
            masks = masks.cuda()
            label = label.cuda()
            masks = masks.unsqueeze(1)

            output_s, cam_refined_s, fea_s = StudentModel(image)
            output_t, cam_refined_t, fea_t = TeacherModel(image)
            class_idx = label.cpu().long().numpy()
            for index, idx in enumerate(class_idx):
                tmp_s = cam_refined_s[index, idx, :, :].unsqueeze(0).unsqueeze(1)
                tmp_t = cam_refined_t[index, idx, :, :].unsqueeze(0).unsqueeze(1)
                if index == 0:
                    cam_refined_class_s = tmp_s
                    cam_refined_class_t = tmp_t
                else:
                    cam_refined_class_s = torch.cat((cam_refined_class_s, tmp_s), dim=0)
                    cam_refined_class_t = torch.cat((cam_refined_class_t, tmp_t), dim=0)
            cam_refined_s = cam_refined_class_s
            cam_refined_t = cam_refined_class_t
            probe_s = torch.softmax(output_s, dim=1)
            probe_t = torch.softmax(output_t, dim=1)

            cam_refined_s = cam_refined_s >= cam_thresh
            cam_refined_t = cam_refined_t >= cam_thresh

            batch_iou_s = single_IOU(cam_refined_s[:, 0, :, :], masks[:, 0, :, :])
            batch_ior_s = single_IOR(cam_refined_s[:, 0, :, :], masks[:, 0, :, :])
            batch_iou_t = single_IOU(cam_refined_t[:, 0, :, :], masks[:, 0, :, :])
            batch_ior_t = single_IOR(cam_refined_t[:, 0, :, :], masks[:, 0, :, :])

            for j in range(len(thresh_TIOU)):
                if batch_iou_s >= thresh_TIOU[j]:
                    TIOU_s[class_idx, j] += 1
                if batch_iou_t >= thresh_TIOU[j]:
                    TIOU_t[class_idx, j] += 1
            for j in range(len(thresh_TIOR)):
                if batch_ior_s >= thresh_TIOR[j]:
                    TIOR_s[class_idx, j] += 1
                if batch_ior_t >= thresh_TIOR[j]:
                    TIOR_t[class_idx, j] += 1
            counts[class_idx] += 1

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

        for idx in range(num_classes):
            for j in range(len(thresh_TIOU)):
                if counts[idx] == 0:
                    TIOU_s[idx, j] = 0.
                    TIOU_t[idx, j] = 0.
                else:
                    TIOU_s[idx, j] = float(TIOU_s[idx, j]) / float(counts[idx])
                    TIOU_t[idx, j] = float(TIOU_t[idx, j]) / float(counts[idx])

        for idx in range(num_classes):
            for j in range(len(thresh_TIOR)):
                if counts[idx] == 0:
                    TIOR_s[idx, j] = 0.
                    TIOR_t[idx, j] = 0.
                else:
                    TIOR_s[idx, j] = float(TIOR_s[idx, j]) / float(counts[idx])
                    TIOR_t[idx, j] = float(TIOR_t[idx, j]) / float(counts[idx])

        return [m_auc_s, all_auc_s, m_acc_s, all_acc_s, m_f1_s, all_f1_s], \
               [m_auc_t, all_auc_t, m_acc_t, all_acc_t, m_f1_t, all_f1_t], \
               [TIOU_s, TIOU_t], \
               [TIOR_s, TIOR_t]


def infer(infer_loader, model, config):
    batch_time = AverageMeter()
    model.eval()

    all_name = []

    with torch.no_grad():
        end = time.time()
        for i, (input, meta) in tqdm(enumerate(infer_loader)):
            name = meta[0]
            all_name = all_name + list(name)
            image = input.squeeze()

            image = image.cuda()

            output, cam_refined, fea = model(image)
            probe = torch.softmax(output, dim=1)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % (config['print_freq'] * config['batchsize']) == 0:
                logging.info('Infer-Cls: [{}/{}]\t'
                             'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(
                    i, len(infer_loader), batch_time=batch_time))

            if i == 0:
                features = fea.cpu().detach().numpy()
                probes = probe.cpu().detach().numpy()
            else:
                features = np.concatenate((features, fea.cpu().detach().numpy()), axis=0)
                probes = np.concatenate((probes, probe.cpu().detach().numpy()), axis=0)

        logging.info('Infer-Cls: features shape: {}'.format(features.shape))
        logging.info('Infer-Cls: probes shape: {}'.format(probes.shape))
        assert features.shape[0] == features.shape[0], 'num error!'
        if not os.path.isdir('./feature'):
            os.makedirs('./feature')
        np.save('./feature/all_features_valid{}.npy'.format(config['valid_fold']), features)
        np.save('./feature/all_probes_valid{}.npy'.format(config['valid_fold']), probes)
        df = pd.DataFrame({'segid': all_name})
        df.to_csv('./feature/all_name_valid{}.csv'.format(config['valid_fold']), index=False)


def single_IOU(pred, target):
    pred_class = pred.data.cpu().contiguous().view(-1)
    target_class = target.data.cpu().contiguous().view(-1)
    pred_inds = pred_class == 1
    target_inds = target_class == 1
    intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
    iou = float(intersection) / float(max(union, 1))
    return iou


def single_IOR(pred, target):
    pred_class = pred.data.cpu().contiguous().view(-1)
    target_class = target.data.cpu().contiguous().view(-1)
    pred_inds = pred_class == 1
    target_inds = target_class == 1
    intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
    iou = float(intersection) / float(max(pred_inds.long().sum().item(), 1))
    return iou


def visual_masks(masks, im_h, im_w):
    mask_vis = masks[0, :, :, :].unsqueeze(0).clone()
    mask_one = torch.zeros((1, im_h, im_w)).cuda()
    mask_one = mask_one + mask_vis[:, 0, :, :]
    mask_one[mask_one >= 1] = 1
    vis_mask1 = mask_one.clone()
    vis_mask2 = mask_one.clone()
    vis_mask3 = mask_one.clone()
    vis_mask1[vis_mask1 == 1] = palette[1][0]
    vis_mask2[vis_mask2 == 1] = palette[1][1]
    vis_mask3[vis_mask3 == 1] = palette[1][2]
    vis_mask1 = vis_mask1.unsqueeze(1)
    vis_mask2 = vis_mask2.unsqueeze(1)
    vis_mask3 = vis_mask3.unsqueeze(1)
    vis_mask = torch.cat((vis_mask1, vis_mask2, vis_mask3), 1)
    return vis_mask

