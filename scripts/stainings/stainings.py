import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm.auto import tqdm
from atp.utils import load_best_model
from omegaconf import DictConfig
from atp.dataset import ATPDataset
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from pandarallel import pandarallel
from PIL import Image


def get_cropped_spheroid(row, channel="BFL", tight=False, root=None):
    if root:
        image = Image.open(f"{root}/{row[channel]}")
    else:
        image = Image.open(f"{row[channel]}")
    xmin, ymin, xmax, ymax = (
        row.xmin * image.width,
        row.ymin * image.height,
        row.xmax * image.width,
        row.ymax * image.height,
    )
    if channel != "BFL" and "cxmin" in row and pd.notna(row.cxmin):
        xmin, ymin, xmax, ymax = (
            row.cxmin * image.width,
            row.cymin * image.height,
            row.cxmax * image.width,
            row.cymax * image.height,
        )

    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    size = max(w, h) // 2
    if not tight and channel != "BFL":
        size = int(size * 1.40)
    # size = 400
    image = image.crop([cx - size, cy - size, cx + size, cy + size])
    # image = image.crop([xmin, ymin, xmax, ymax])
    return image


def intensity(row, channel, root=None):
    image = get_cropped_spheroid(row, channel=channel, root=root)
    w, h = int(image.width * 0.1), int(image.height * 0.1)
    data = np.array(image)
    background = np.concatenate(
        [
            data[:h, :w].ravel(),
            data[:h, -w:].ravel(),
            data[-h:, :w].ravel(),
            data[-h:, -w:].ravel(),
        ]
    )
    background = np.quantile(background, 0.5)
    data = data - background
    data = data.clip(0, 255).astype(np.uint8)
    area = (data > 0).sum()
    return data.sum() / area


pandarallel.initialize(progress_bar=True)
pd.set_option("display.precision", 3)
pd.options.display.float_format = "{:.3f}".format
from atp.preprocess import normalize_by_compound
from scipy.stats import pearsonr


root = "data/images_chromalive/"
df = pd.read_csv("data/data_chromalive.csv")
df["yellow488"] = df.parallel_apply(
    lambda x: intensity(x, "yellow488_image", root), axis=1
)
df["red488"] = df.parallel_apply(lambda x: intensity(x, "red488_image", root), axis=1)
df["green561"] = df.parallel_apply(
    lambda x: intensity(x, "green561_image", root), axis=1
)


device = torch.device("mps:0")
colors = ["yellow488", "red488", "green561", "y_pred"]

# load model
model_id = "data/model_fcdnoe3o.ckpt"  # trained on A
model = load_best_model(run_path=model_id)
config = DictConfig(model.config)
transforms = A.Compose(
    [
        A.Resize(*config.datamodule.resolution, interpolation=5),
        A.Normalize(0.5, 0.5),
        ToTensorV2(),
    ]
)

# get latent vectors of thorlabs images
root = "./data/images_labeled_cropped_0.2/"
df4pca = pd.read_csv("./data/data_labeled.csv").query('microscope == "thorlabs"')
df4pca = normalize_by_compound(df4pca)  # normalize to calculate viabilty
ds = ATPDataset(df4pca, root, transforms)
dl = DataLoader(ds, batch_size=8)
model = model.to(device)
df4pca_features = []
y_pred4pca = []
with torch.no_grad():
    for batch in tqdm(dl):
        x = batch["image"].to(device)
        df4pca_features.append(model.encoder(x).cpu())
        y_pred4pca.append(model(x).cpu())
        torch.cuda.empty_cache()
df4pca_features = torch.concat(df4pca_features).numpy()
y_pred4pca = torch.concat(y_pred4pca).numpy().ravel()

df4pca["y_pred"] = y_pred4pca

# get latent vectors of brightfield images of stained spheroids
df["location_in_bucket"] = df.BFL
df["microscope"] = "thorlabs"
root = "data/images_chromalive_cropped_0.2/"
ds = ATPDataset(df, root, transforms)
dl = DataLoader(ds, batch_size=8)
device = torch.device("cpu:0")
model = model.to(device)
encoder_features = []
y_pred = []
with torch.no_grad():
    for batch in tqdm(dl):
        x = batch["image"].to(device)
        encoder_features.append(model.encoder(x).cpu())
        y_pred.append(model(x).cpu())
        torch.cuda.empty_cache()
encoder_features = torch.concat(encoder_features).numpy()
y_pred = torch.concat(y_pred).numpy().ravel()
df["encoder_features"] = encoder_features.tolist()
df["y_pred"] = y_pred


# calculate correlations
features = np.stack(df.encoder_features)
pca = PCA()
components4pca = pca.fit_transform(df4pca_features)
components = pca.transform(features)
df["pca_components"] = components.tolist()

d_pca = pd.DataFrame()
for c in colors:
    q = df[df[c].notna()]
    corr = [pearsonr(q[c], components[q.index, i]) for i in range(10)]
    d_pca[c + "_corr"] = [c.statistic for c in corr]
    d_pca[c + "_pvalue"] = [c.pvalue for c in corr]

corr = [pearsonr(df4pca["value"], components4pca[:, i]) for i in range(10)]
d_pca["y_true_corr"] = [c.statistic for c in corr]
d_pca["y_true_pvalue"] = [c.pvalue for c in corr]

print("max correlation to 10 pca components")
print(d_pca[[c for c in d_pca.columns if "pvalue" not in c]].abs().max())


# correlations vs y_pred, y_true
print("vs y_pred")
for c in colors:
    q = df[df[c].notna()]
    corr = pearsonr(q[c], q["y_pred"])
    print(c, f"{np.abs(corr.statistic):.3f}, pvalue: {corr.pvalue:.5f}")

print("\nvs atp")
q = df[df.value.notna()]
for c in colors:
    qq = q[q[c].notna()]
    corr = pearsonr(qq[c], qq["value"])
    print(c, f"{np.abs(corr.statistic):.3f}, pvalue: {corr.pvalue:.5f}")

print("top 10 PCs")
print(d_pca[:10].abs())

print("orthogonal PC to viability from top 10 PCs")
print(d_pca[:10].abs().query("y_true_corr < 0.1"))
