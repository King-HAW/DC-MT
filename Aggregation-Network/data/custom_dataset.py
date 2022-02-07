import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import itertools
from torch.utils.data.sampler import Sampler


def relabel_dataset(dataset, labels):
    labeled_idxs = dataset.mode_file['SegID'].isin(labels)
    labeled_idxs = labeled_idxs[labeled_idxs == True].index.values.tolist()

    assert len(labeled_idxs) == len(labels), 'Labeled num do not match'
    unlabeled_idxs = sorted(set(range(len(dataset.mode_file))) - set(labeled_idxs))

    return labeled_idxs, unlabeled_idxs


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            secondary_batch + primary_batch
            for (secondary_batch, primary_batch)
            in  zip(grouper(secondary_iter, self.secondary_batch_size),
                    grouper(primary_iter, self.primary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class CustomDataset(data.Dataset):
    def __init__(self, name, fea_path, pro_path, query_index_path, mode_file_path, folds):
        self.name = name

        # Load Data
        self.fea_npy = np.load(fea_path).astype(np.float16)
        self.pro_npy = np.load(pro_path).astype(np.float16)
        self.query = pd.read_csv(query_index_path)
        self.mode_file = pd.read_csv(mode_file_path)

        if folds:
            self.mode_file = self.mode_file[self.mode_file['Fold'].isin(folds)]
            self.mode_file = self.mode_file.reset_index(drop=True)

        self.patient_list = list(self.mode_file['SegID'].unique())

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, index):
        segid = self.patient_list[index]
        patient_query_df = self.query[self.query['SegID'].isin([segid])]
        all_index = patient_query_df['fea_index'].to_list()
        if len(all_index) >= 24:
            start = (len(all_index) - 24) // 2
            all_index = all_index[start:start+24]
            fea_tmp = [self.fea_npy[tmp, :] for tmp in all_index]
            pro_tmp = [self.pro_npy[tmp, :] for tmp in all_index]
            fea_tmp_zero = np.asarray(fea_tmp)
            pro_tmp_zero = np.asarray(pro_tmp)

        else:
            fea_tmp = [self.fea_npy[tmp, :] for tmp in all_index]
            pro_tmp = [self.pro_npy[tmp, :] for tmp in all_index]
            fea_tmp = np.asarray(fea_tmp)
            pro_tmp = np.asarray(pro_tmp)
            fea_tmp_zero = np.zeros((24, fea_tmp.shape[-1]))
            pro_tmp_zero = np.zeros((24, pro_tmp.shape[-1]))
            start_idx = (24 - len(all_index)) // 2
            fea_tmp_zero[start_idx:start_idx+len(all_index), :] = fea_tmp
            pro_tmp_zero[start_idx:start_idx+len(all_index), :] = pro_tmp

        patient_mode_df = self.mode_file[self.mode_file['SegID'].isin([segid])]
        label = patient_mode_df['subject_label'].tolist()
        label = np.asarray(label)

        fea = torch.from_numpy(fea_tmp_zero).float()
        pro = torch.from_numpy(pro_tmp_zero).float()
        label = torch.from_numpy(label).long()

        return (fea, pro), (fea, pro), label, segid

