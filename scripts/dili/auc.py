import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve
import click


@click.command()
@click.argument("df_path")
def calc_auc(df_path):
    proctor = pd.read_csv("./data/proctor.csv")
    proctor_mos = (proctor.ic50_float / proctor.cmax).map(
        lambda x: 1e6 if x == np.inf else x
    )
    fpr, tpr, thresholds = roc_curve(
        y_true=proctor.binary_label, y_score=proctor_mos, pos_label=0
    )
    print("proctor", auc(fpr, tpr))

    ic50 = pd.read_csv(df_path).query("day == 7")
    our_mos = (ic50.ic50_calc / ic50.cmax).map(lambda x: 1e6 if x == np.inf else x)
    fpr, tpr, thresholds = roc_curve(
        y_true=ic50.binary_label, y_score=our_mos, pos_label=0
    )
    print("ours", auc(fpr, tpr))


if __name__ == "__main__":
    calc_auc()
