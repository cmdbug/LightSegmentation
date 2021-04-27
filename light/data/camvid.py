import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as data
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import transforms
from light.utils.visualize import get_color_pallete

__all__ = ['CamVidSegmentation']


class CamVidSegmentation(data.Dataset):
    BASE_DIR = 'camvid'
    NUM_CLASS = 13

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        ('road_marking', (255, 69, 0)),
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        ('unlabeled', (0, 0, 0))
    ])

    def __init__(self, root='G:/dataset/camvid', split='train', mode=None, transform=None,
                 base_size=520, crop_size=480, **kwargs):
        super(CamVidSegmentation, self).__init__()
        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")
        # self._key = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])  # 天呐，这坑了半天，但又学了几招^_^
        self._key = np.arange(-1, CamVidSegmentation.NUM_CLASS + 1)
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


def pil_loader(data_path, label_path):
    """Loads a sample and label image given their path as PIL images.

    Keyword arguments:
    - data_path (``string``): The filepath to the image.
    - label_path (``string``): The filepath to the ground-truth image.

    Returns the image and the label as PIL images.

    """
    data = Image.open(data_path)
    label = Image.open(label_path)

    return data, label


def get_files(folder, name_filter=None, extension_filter=None):
    """Helper function that returns the list of files in a specified folder
    with a specified extension.

    Keyword arguments:
    - folder (``string``): The path to a folder.
    - name_filter (```string``, optional): The returned files must contain
    this substring in their filename. Default: None; files are not filtered.
    - extension_filter (``string``, optional): The desired file extension.
    Default: None; files are not filtered

    """
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    # Filename filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files that do not
    # contain "name_filter"
    if name_filter is None:
        # This looks hackish...there is probably a better way
        name_cond = lambda filename: True
    else:
        name_cond = lambda filename: name_filter in filename

    # Extension filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files whose extension
    # is not "extension_filter"
    if extension_filter is None:
        # This looks hackish...there is probably a better way
        ext_cond = lambda filename: True
    else:
        ext_cond = lambda filename: filename.endswith(extension_filter)

    filtered_files = []

    # Explore the directory tree to get files that contain "name_filter" and
    # with extension "extension_filter"
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)

    return filtered_files


def _get_city_pairs(folder, split='train'):
    train_folder = 'train'
    train_lbl_folder = 'trainannot'

    # Validation dataset root folders
    val_folder = 'val'
    val_lbl_folder = 'valannot'

    # Test dataset root folders
    test_folder = 'test'
    test_lbl_folder = 'testannot'

    # Images extension
    img_extension = '.png'

    if split in 'train':
        img_paths = get_files(os.path.join(folder, train_folder), extension_filter=img_extension)
        mask_paths = get_files(os.path.join(folder, train_lbl_folder), extension_filter=img_extension)
    elif split in 'val':
        img_paths = get_files(os.path.join(folder, val_folder), extension_filter=img_extension)
        mask_paths = get_files(os.path.join(folder, val_lbl_folder), extension_filter=img_extension)
    else:
        img_paths = get_files(os.path.join(folder, test_folder), extension_filter=img_extension)
        mask_paths = get_files(os.path.join(folder, test_lbl_folder), extension_filter=img_extension)

    return img_paths, mask_paths


if __name__ == '__main__':
    matplotlib.use('TKAgg')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.style.use(['fast'])
    plt.rcParams['figure.facecolor'] = 'gray'

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    # Create Dataset
    trainset = CamVidSegmentation(split='train', transform=input_transform)
    # Create Training Loader
    train_data = data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

    images, masks = trainset.__getitem__(0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    # masks = np.expand_dims(masks, 1)
    mask = masks.numpy()
    rgb_mask = get_color_pallete(mask, "camvid")
    ax2.imshow(rgb_mask)
    plt.show()

    # for images, masks in train_data:
    #     # print(images.shape, masks.shape)
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    #     ax1.imshow(np.transpose(images[0], (1, 2, 0)))
    #     # masks = np.expand_dims(masks, 1)
    #     mask = masks[0].numpy()
    #     rgb_mask = get_color_pallete(mask, "camvid")
    #     ax2.imshow(rgb_mask)
    #     plt.show()
    #     break
