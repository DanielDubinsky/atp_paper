import pandas as pd
from PIL import Image
from torchvision.transforms.functional import crop
import os
import click
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


@click.command()
@click.argument("df_path")
@click.argument("root")
@click.argument("dst_dir")
@click.option("--size", default=0.2)
@click.argument("cols", nargs=-1)
def crop_spheroids(df_path, root, dst_dir, size, cols):
    df = pd.read_csv(df_path)
    to_crop = df[df.xmin.notna()].reset_index(drop=True)
    no_spheroids = df[df.xmin.isna()]
    if len(cols) == 0:
        cols = ["location_in_bucket"]
    # print('no spheroid:', len(no_spheroids))
    assert len(no_spheroids) == 0
    for col in cols:
        print(col)

        def crop_around_spheroid(
            row,
        ):
            d = dst_dir + "/" + row[col]
            if os.path.exists(d):
                return
            image = Image.open(root + "/" + row[col])
            s = min(image.height, image.width) * size
            xmin, ymin, xmax, ymax = row[["xmin", "ymin", "xmax", "ymax"]]
            xmin, ymin, xmax, ymax = (
                xmin * image.width,
                ymin * image.height,
                xmax * image.width,
                ymax * image.height,
            )
            center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
            cropped_image = crop(image, center_y - s, center_x - s, s * 2, s * 2)

            os.makedirs(os.path.dirname(d), exist_ok=True)
            cropped_image.save(d)

        to_crop.parallel_apply(crop_around_spheroid, axis=1)


if __name__ == "__main__":
    crop_spheroids()
