from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
import random
import numpy as np
import scipy.io as sio


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map

    # fast create discrete map
    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h - 1] * num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w - 1] * num_gt).astype(int))
    p_index = torch.from_numpy(p_h * im_width + p_w).to(torch.int64)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index,
                                                                  src=torch.ones(im_width * im_height)).view(im_height,
                                                                                                             im_width).numpy()

    ''' slow method
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''
    assert np.sum(discrete_map) == num_gt
    return discrete_map


class Base(data.Dataset):
    def __init__(self, root_path, crop_size, downsample_ratio=8):

        self.root_path = root_path
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def train_transform(self, img, keypoints, gauss_im):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = TF.crop(img, i, j, h, w)
        gauss_im = TF.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = TF.hflip(img)
                gauss_im = TF.hflip(gauss_im)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = TF.hflip(img)
                gauss_im = TF.hflip(gauss_im)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), gauss_im, torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()


class Crowd_TC(Base):
    def __init__(self, root_path, crop_size, downsample_ratio=8, method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")

        self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))

        print('number of img [{}]: {}'.format(method, len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        name = os.path.basename(img_path).split('.')[0]
        gd_path = os.path.join(self.root_path, 'ground_truth', 'GT_{}.mat'.format(name))
        img = Image.open(img_path).convert('RGB')
        mat_data = sio.loadmat(gd_path)
        if 'image_info' in mat_data:
            keypoints = mat_data['image_info'][0][0][0][0][0]
        elif 'annPoints' in mat_data:
            keypoints = mat_data['annPoints']
        else:
            raise KeyError(
                f"Key 'image_info' or 'annPoints' not found in {gd_path}. Found keys: {list(mat_data.keys())}")
        gauss_path = os.path.join(self.root_path, 'ground_truth', '{}_densitymap.npy'.format(name))
        gauss_im = torch.from_numpy(np.load(gauss_path)).float()
        # import pdb;pdb.set_trace()
        # print("label {}", item)

        if self.method == 'train':
            return self.train_transform(img, keypoints, gauss_im)
        elif self.method in ['val', 'test']:
            # Validation / test: unify with training preprocessing
            # 1) Resize image and gauss_im to 512x512 with Bicubic
            wd, ht = img.size
            target_size = 512
            rr_w = 1.0 * target_size / wd
            rr_h = 1.0 * target_size / ht
            img = img.resize((target_size, target_size), Image.BICUBIC)
            gauss_im = F.interpolate(
                gauss_im.unsqueeze(0).unsqueeze(0),
                size=(target_size, target_size),
                mode='bicubic',
                align_corners=False
            ).squeeze(0).squeeze(0)

            keypoints = keypoints.astype(np.float32)
            if len(keypoints) > 0:
                keypoints[:, 0] = keypoints[:, 0] * rr_w
                keypoints[:, 1] = keypoints[:, 1] * rr_h

            wd, ht = target_size, target_size

            # 2) Center crop to 256x256 for deterministic eval
            crop_h = self.c_size
            crop_w = self.c_size
            i = max((ht - crop_h) // 2, 0)
            j = max((wd - crop_w) // 2, 0)
            img = TF.crop(img, i, j, crop_h, crop_w)
            gauss_im = TF.crop(gauss_im, i, j, crop_h, crop_w)

            # 3) Shift keypoints into cropped patch and keep only inside
            if len(keypoints) > 0:
                keypoints = keypoints - [j, i]
                idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= crop_w) * \
                           (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= crop_h)
                keypoints = keypoints[idx_mask]

            img = self.trans(img)
            # Return image tensor, updated count, name, and cropped gauss_im
            return img, len(keypoints), name, gauss_im

    def train_transform(self, img, keypoints, gauss_im):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # First resize image and gauss_im to 512x512 using Bicubic interpolation
        target_size = 512
        rr_w = 1.0 * target_size / wd
        rr_h = 1.0 * target_size / ht
        img = img.resize((target_size, target_size), Image.BICUBIC)
        gauss_im = F.interpolate(gauss_im.unsqueeze(0).unsqueeze(0), size=(target_size, target_size), mode='bicubic',
                                 align_corners=False).squeeze(0).squeeze(0)
        keypoints = keypoints.astype(np.float32)
        if len(keypoints) > 0:
            keypoints[:, 0] = keypoints[:, 0] * rr_w
            keypoints[:, 1] = keypoints[:, 1] * rr_h
        wd, ht = target_size, target_size
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        # Then randomly crop 256x256 patches for training
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = TF.crop(img, i, j, h, w)
        gauss_im = TF.crop(gauss_im, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = TF.hflip(img)
                gauss_im = TF.hflip(gauss_im)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = TF.hflip(img)
                gauss_im = TF.hflip(gauss_im)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)
        # import pdb;pdb.set_trace()

        return self.trans(img), gauss_im, torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()


class Base_UL(data.Dataset):
    def __init__(self, root_path, crop_size, downsample_ratio=8):
        self.root_path = root_path
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def train_transform_ul(self, img):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = TF.crop(img, i, j, h, w)

        if random.random() > 0.5:
            img = TF.hflip(img)

        return self.trans(img)


class Crowd_UL_TC(Base_UL):
    def __init__(self, root_path, crop_size, downsample_ratio=8, method='train_ul'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train_ul']:
            raise Exception("not implement")

        self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))
        print('number of img [{}]: {}'.format(method, len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        name = os.path.basename(img_path).split('.')[0]
        img = Image.open(img_path).convert('RGB')
        # print("un_label {}", item)

        return self.train_transform_ul(img)

    def train_transform_ul(self, img):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)

        assert st_size >= self.c_size, print(wd, ht)

        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = TF.crop(img, i, j, h, w)
        if random.random() > 0.5:
            img = TF.hflip(img)

        return self.trans(img), 1

