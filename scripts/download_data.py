from tqdm.auto import tqdm
import shutil
from pathlib import Path
import os
from quris_ml.dataset.vision.dataset import VisionDataset
from joblib import Parallel, delayed
from PIL import Image

ds = VisionDataset.from_wandb("atp_paper/dataset:latest")

root = Path("/home/danieldubinsky/atp_paper/data/")
df = ds._df.copy()


def cp(local_path):
    idx = local_path.find("microscopy_data")
    dst = root / Path(local_path[idx:])
    os.makedirs(dst.parent, exist_ok=True)
    # shutil.copy(local_path, dst)

    # crop center
    image = Image.open(local_path)
    w, h = image.width, image.height
    margin = (w - h) // 2

    # cropped_image = crop(image, center_y - size, center_x - size, size * 2, size * 2)
    cropped_image = image.crop([margin, 0, margin + h, h])
    cropped_image.save(dst)

    return local_path[idx:]


new_paths = Parallel(n_jobs=10)(
    delayed(cp)(local_path) for local_path in tqdm(df.local_path)
)
df["local_path"] = new_paths
df["split"] = "train_val"
df.loc[(df.location == "A") & (df.study == "ms12"), "split"] = "test"
df.loc[(df.location == "B") & (df.study == "ms17"), "split"] = "test"
df.to_csv(root / "data.csv", index=False)
