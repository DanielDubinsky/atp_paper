import pandas as pd
import numpy as np
import cv2
from torchvision.transforms.functional import crop
import os
from pathlib import Path
import click
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


@click.command()
@click.argument("df_path")
@click.argument("root")
@click.argument("dst_dir")
@click.option("--size", default=256)
@click.argument("cols", nargs=-1)
def crop_spheroids(df_path, root, dst_dir, size, cols):
    df = pd.read_csv(df_path)
    if len(cols) == 0:
        cols = ["location_in_bucket"]
    for col in cols:
        print(col)

        def resize(
            row,
        ):
            d = dst_dir + "/" + row[col]
            if os.path.exists(d):
                return
            image = cv2.imread(root + row[col], cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (size, size), interpolation=5)

            os.makedirs(os.path.dirname(d), exist_ok=True)
            cv2.imwrite(d, image)

        df.parallel_apply(resize, axis=1)


if __name__ == "__main__":
    crop_spheroids()
