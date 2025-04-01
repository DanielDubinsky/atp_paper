import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import os
from torchvision.transforms.functional import pil_to_tensor
from atp.dataset import ATPDataset
from atp.preprocess import normalize_by_compound
from PIL import Image

classic_features = ["pc", "curvature", "brightness"]


def get_contour(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        arclengths = [cv2.arcLength(c, True) for c in contours]
        idx = np.argmax(arclengths)
        area = cv2.contourArea(contours[idx])
        equi_diameter = np.sqrt(4 * area / np.pi)
        circum = equi_diameter * np.pi
        return contours[idx], arclengths[idx], circum
    else:
        return None, None, None


def curvature(contour):
    contour = contour.reshape(-1, 2)

    x_t = np.gradient(contour[:, 0])
    y_t = np.gradient(contour[:, 1])

    # vel = np.array([ [x_t[i], y_t[i]] for i in range(x_t.size)])
    # speed = np.sqrt(x_t * x_t + y_t * y_t)
    # tangent = np.array([1/speed] * 2).transpose() * vel
    # ss_t = np.gradient(speed)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)

    curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t) ** 1.5
    return np.nanmean(curvature_val)


def get_brightness_feature(row, mask, image):
    h, w = image.size()
    xmin, ymin, xmax, ymax = (
        int(w * row.xmin),
        int(h * row.ymin),
        int(w * row.xmax),
        int(h * row.ymax),
    )
    cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
    fsize = h // 20
    foreground = image[cy - fsize : cy + fsize, cx - fsize : cx + fsize].mean()

    # get the four corners of the image as background
    bsize = h // 40
    background = torch.stack(
        [
            image[ymin : ymin + bsize, xmin : xmin + bsize],
            image[ymax - bsize : ymax, xmin : xmin + bsize],
            image[ymin : ymin + bsize, xmax - bsize : xmax],
            image[ymax - bsize : ymax, xmax - bsize : xmax],
        ]
    ).median()
    return (foreground / background).item()


root = "./data/images_labeled_256/"
mask_root = "./data/images_labeled_masks/"
src_df_path = "./data/data_labeled.csv"
dst_df_path = "./data/data_labeled_w_classic_features.csv"
df = pd.read_csv(src_df_path)
df = df[df.xmin.notna()]
transforms = A.Compose([A.Resize(256, 256, interpolation=5), ToTensorV2()])
ds = ATPDataset(df, root, transforms)


from joblib import Parallel, delayed

# Parallel(n_jobs=6)(delayed(proc_img)(d['image'][i]) for i in tqdm(range(len(d['image'])), desc='images'))

df = ds.df.copy()
# df['contour_norm']


# for i in tqdm(range(len(df))):
def proc(i):
    row = df.iloc[i]
    image = (ds[i]["image"] / 255).to(torch.float)
    mask = pil_to_tensor(
        Image.open(mask_root + row.location_in_bucket).resize((256, 256))
    ).to(torch.uint8)
    x = mask.numpy()[0]
    contour, perimeter, circum = get_contour(x)
    brightness = get_brightness_feature(row, mask, image[0])

    d = {}
    if contour is not None:
        d["pc"] = circum / perimeter
        d["curvature"] = curvature(contour)
        d["brightness"] = brightness
    else:
        d["pc"] = pd.NA
        d["curvature"] = pd.NA
        d["brightness"] = pd.NA
    return d


ret = Parallel(n_jobs=8)(delayed(proc)(i) for i in tqdm(range(len(df))))
# ret = [proc(i) for i in tqdm(range(len(df)))]
ret_df = pd.DataFrame(ret)
ret_df.index = df.index
df = pd.concat([df, ret_df], axis=1)

for c in classic_features:
    df = normalize_by_compound(df, src_col=c, dst_col=c)
    # df = set_control(df, src_col=c)
# set control features
# df_norm = set_control(df)
# df_norm.control = df_norm.control.map(lambda x: x[len(x) // 2])
# df_control = pd.DataFrame(df_norm.control.map(lambda x: df_norm.loc[x]).tolist())
# df_control.index = df_norm.index
# df = df.join(df_control[classic_features], rsuffix='_control')

df.to_csv(dst_df_path, index=False)
