import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm.notebook import tqdm
from atp.utils import load_best_model
from omegaconf import DictConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2
from atp.dataset import ATPDataset
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
pd.set_option("display.precision", 3)
pd.options.display.float_format = "{:.3f}".format
from atp.preprocess import normalize_by_compound


df = pd.read_csv("data/data_final_chromalive.csv")
device = torch.device("cuda:0")
colors = ["yellow488", "red488", "green561", "y_pred"]

# load model
model_id = "quris/atp_paper_final2_cropped_dropout0.2/fcdnoe3o"  # trained on A
model = load_best_model(run_path=model_id)
config = DictConfig(model.config)
transforms = A.Compose(
    [A.Resize(*config.datamodule.resolution, interpolation=5), A.Normalize(0.5, 0.5), ToTensorV2()]
)

# get latent vectors of thorlabs images
root = "data/final_cropped_0.2/"
df4pca = pd.read_csv("data/data_final.csv").query('microscope == "thorlabs"')
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
root = "data/final_chromalive_cropped_0.2/"
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

df.to_csv("data/data_final_chromalive.csv", index=False)
