from PIL import Image, ImageOps, ImageFilter
import platform, os
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import torch
from torch.nn import init
from datetime import datetime
import argparse
import shutil
from scipy import ndimage
from scipy.ndimage.interpolation import zoom


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)

    return image, label


class TrainSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, base_size=512, crop_size=480, suffix='.png'):
        super(TrainSetLoader, self).__init__()

        self._items = img_id
        self.masks = dataset_dir + '/' + 'masks'
        self.images = dataset_dir + '/' + 'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    # def _sync_transform(self, img, mask):
    #     # random mirror
    #     if random.random() < 0.5:
    #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #         mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    #     crop_size = self.crop_size
    #     # random scale (short edge)
    #     long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
    #     w, h = img.size
    #     if h > w:
    #         oh = long_size
    #         ow = int(1.0 * w * long_size / h + 0.5)
    #         short_size = ow
    #     else:
    #         ow = long_size
    #         oh = int(1.0 * h * long_size / w + 0.5)
    #         short_size = oh
    #     img = img.resize((ow, oh), Image.BILINEAR)
    #     mask = mask.resize((ow, oh), Image.NEAREST)
    #     # pad crop
    #     if short_size < crop_size:
    #         padh = crop_size - oh if oh < crop_size else 0
    #         padw = crop_size - ow if ow < crop_size else 0
    #         img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #         mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
    #     # random crop crop_size
    #     w, h = img.size
    #     x1 = random.randint(0, w - crop_size)
    #     y1 = random.randint(0, h - crop_size)
    #     img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    #     mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    #     # gaussian blur as in PSP
    #     if random.random() < 0.5:
    #         img = img.filter(ImageFilter.GaussianBlur(
    #             radius=random.random()))
    #     # final transform
    #     img, mask = np.array(img, dtype=np.float32), np.array(mask, dtype=np.float32)
    #     return img, mask

    def _sync_transform(self, img, mask):

        img, mask = np.array(img), np.array(mask)
        if random.random() > 0.5:
            img, mask = random_rot_flip(img, mask)
        elif random.random() > 0.5:
            img, mask = random_rotate(img, mask)

        x, y, _ = img.shape
        if x != self.base_size or y != self.crop_size:
            img = zoom(img, (self.base_size / x, self.crop_size / y, 1), order=3)

        h, w = mask.shape
        if h != self.base_size or w != self.crop_size:
            mask = zoom(mask, (self.base_size / h, self.crop_size / w), order=3)

        img, mask = np.array(img, dtype=np.float32), np.array(mask, dtype=np.float32)

        return img, mask


    def __getitem__(self, idx):

        img_id = self._items[idx]
        img_path = self.images + '/' + img_id + self.suffix
        label_path = self.masks + '/' + img_id + self.suffix

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path).convert('L')


        # synchronized transform
        img, mask = self._sync_transform(img, mask)

        # mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0
        mask = mask.astype('float32') / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, torch.from_numpy(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)


class TestSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, base_size=512, crop_size=480, suffix='.png'):
        super(TestSetLoader, self).__init__()
        self._items = img_id
        self.masks = dataset_dir + '/' + 'masks'
        self.images = dataset_dir + '/' + 'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        # final transform
        img, mask = np.array(img, dtype=np.float32), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    # def _testval_sync_transform(self, img, mask):
    #
    #     img, mask = np.array(img), np.array(mask)
    #
    #     x, y, _ = img.shape
    #     h, w = mask.shape
    #
    #     if x != self.base_size or y != self.crop_size:
    #         img = zoom(img, (self.base_size / x, self.crop_size / y, 1), order=3)
    #     if h != self.base_size or w != self.crop_size:
    #         mask = zoom(mask, (self.base_size / h, self.crop_size / w), order=3)
    #
    #     img, mask = np.array(img, dtype=np.float32), np.array(mask, dtype=np.float32)
    #
    #     return img, mask


    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path = self.images + '/' + img_id + self.suffix  # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks + '/' + img_id + self.suffix
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)
        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)

        # mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0
        mask = mask.astype('float32') / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, torch.from_numpy(mask), img_id  # img_id[-1]

    def __len__(self):
        return len(self._items)


def load_dataset(root):
    train_txt = root + '/' + 'train.txt'
    test_txt = root + '/' + 'test_vol.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids, val_img_ids, test_txt


# ⚪？
def save_ckpt(state, save_path, filename):
    torch.save(state, os.path.join(save_path, filename))


def save_train_log(args, save_dir):
    dict_args = vars(args)
    args_key = list(dict_args.keys())
    args_value = list(dict_args.values())
    with open('result/%s/train_log.txt' % save_dir, 'w') as f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')
    return


def save_model_and_result(dt_string, epoch, train_loss, test_loss, best_iou, recall, precision, save_mIoU_dir,
                          save_other_metric_dir):
    with open(save_mIoU_dir, 'a') as f:
        f.write('{} - {:04d}: - train_loss: {:04f}: - test_loss: {:04f}: mIoU {:.4f}\n'.format(dt_string, epoch,
                                                                                                     float(train_loss),
                                                                                                     float(test_loss),
                                                                                                     float(best_iou)))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')

def save_model(mean_IOU, nIoU, IoU, best_miou, best_nIoU, best_IoU, save_prefix, train_loss, test_loss, recall, precision, epoch, net):

    save_mIoU_dir = save_prefix + '/' + '_best_IoU_IoU.log'
    # print(save_mIoU_dir)
    save_other_metric_dir = save_prefix + '/' + '_best_IoU_other_metric.log'
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    save_model_and_result(dt_string, epoch, train_loss, test_loss, mean_IOU,
                          recall, precision, save_mIoU_dir, save_other_metric_dir)
    # save_model_and_result(dt_string, epoch, train_loss, test_loss, mean_IOU,
    #                       save_mIoU_dir, save_other_metric_dir)
    if mean_IOU > best_miou:
        best_miou = mean_IOU
        # print("best_mIoU： %f, epoch: %s", best_iou, str(epoch))
        save_mode_path = os.path.join(save_prefix, 'epoch_best' + '.pth')
        torch.save(net, save_mode_path)
    if nIoU > best_nIoU:
        best_nIoU = nIoU
        save_mode_path = os.path.join(save_prefix, 'epoch_best_nIou' + '.pth')
        torch.save(net, save_mode_path)
    if IoU > best_IoU:
        best_IoU = IoU

    return best_miou, best_nIoU, best_IoU
