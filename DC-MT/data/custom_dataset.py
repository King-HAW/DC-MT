import os
import torch
import torch.utils.data as data
import numpy as np
import h5py as h5
import pandas as pd
import itertools
import cv2
import SimpleITK as sitk
from copy import deepcopy
from torch.utils.data.sampler import Sampler

label_to_num = {
    'BackGround':        0,
    'femur':             1,
    'femur_cartilage':   2,
    'L_meniscus':        3,
    'R_meniscus':        4,
    'patella':           5,
    'patella_cartilage': 6,
}
num_to_label = {v: k for k, v in label_to_num.items()}


def relabel_dataset(dataset, labels):
    labeled_idxs = dataset.path_df.Path.isin(labels)
    labeled_idxs = labeled_idxs[labeled_idxs == True].index.values.tolist()

    assert len(labeled_idxs) == len(labels), 'Labeled num do not match'
    unlabeled_idxs = sorted(set(range(len(dataset.path_df))) - set(labeled_idxs))

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


def get_labeled_mask(masks, label):
    masks[masks == label_to_num['femur_cartilage']] = 9
    masks = np.where(masks < 8, 0, masks)
    masks[masks == 9] = 1

    ## Dilation
    kernel_dila = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
    masks = cv2.dilate(masks, kernel_dila)

    return masks


class CustomDataset(data.Dataset):
    def __init__(self, name, data_path, fold_file, folds, transform=None):
        self.name = name
        self.transform = transform

        # Load DataFrame
        self.path_df = pd.read_csv(fold_file)
        self.path_df['ImagePath'] = data_path + self.path_df['Path']

        if folds:
            self.path_df = self.path_df[self.path_df.Fold.isin(folds)]
            self.path_df = self.path_df.reset_index(drop=True)

        pass

    def __len__(self):
        return len(self.path_df)

    def __getitem__(self, index):
        row = self.path_df.iloc[index]
        with h5.File(row.ImagePath, 'r') as f:
            image = f['image'][()]
            masks = f['masks'][()]
            label = f['label'][()]

        flags = np.zeros(1, dtype=np.float32)

        if label != 0:
            flags[0] = 1

        # Normalization
        min_value = np.percentile(image, 0.1)
        max_value = np.percentile(image, 99.9)
        image[image > max_value] = max_value
        image[image < min_value] = min_value  # -outliers
        norm_image = (image - min_value) / (max_value - min_value)
        image = np.array([norm_image, norm_image, norm_image]).transpose((1, 2, 0))

        if self.transform:
            augmented1 = self.transform(image=image, mask=masks)
            image1 = augmented1['image']
            masks1 = augmented1['mask']

            augmented2 = self.transform(image=image, mask=masks)
            image2 = augmented2['image']
            masks2 = augmented2['mask']

        masks1 = get_labeled_mask(masks1, label)
        masks2 = get_labeled_mask(masks2, label)

        image1 = image1.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))
        label = np.array(label)

        image1 = torch.from_numpy(image1).float()
        masks1 = torch.from_numpy(masks1).float()
        image2 = torch.from_numpy(image2).float()
        masks2 = torch.from_numpy(masks2).float()
        label = torch.from_numpy(label).long()
        flags = torch.from_numpy(flags).float()

        return (image1, masks1), (image2, masks2), label, flags, row.Path


class InferDataset(data.Dataset):
    def __init__(self, name, data_path, fold_file, transform=None):
        self.name = name
        self.transform = transform

        # Load DataFrame
        self.path_df = pd.read_csv(fold_file)
        self.path_df['ImagePath'] = data_path + self.path_df['SegID']

    def __len__(self):
        return len(self.path_df)

    def __getitem__(self, index):
        row = self.path_df.iloc[index]
        img = sitk.ReadImage(os.path.join(row.ImagePath, 'org.mha'))
        image = sitk.GetArrayFromImage(img)
        image = np.transpose(image, (1, 2, 0))
        image_list = []

        for slice_index in range(image.shape[-1]):
            slice_image = image[:, :, slice_index]

            # Normalization
            min_value = np.percentile(slice_image, 0.1)
            max_value = np.percentile(slice_image, 99.9)
            slice_image[slice_image > max_value] = max_value
            slice_image[slice_image < min_value] = min_value  # -outliers
            norm_image = (slice_image - min_value) / (max_value - min_value)
            norm_slice_image = np.array([norm_image, norm_image, norm_image]).transpose((1, 2, 0))

            if self.transform:
                augmented = self.transform(image=norm_slice_image)
                norm_slice_image = augmented['image']

            norm_slice_image = norm_slice_image.transpose((2, 0, 1))
            image_list.append(np.expand_dims(norm_slice_image, axis=0))

        final_image = np.concatenate(image_list, axis=0)
        image = torch.from_numpy(final_image).float()

        return image, [row['SegID'], img.GetOrigin(), img.GetSpacing(), img.GetDirection()]
