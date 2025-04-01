import pandas as pd
from typing import Literal

import pandas as pd
from typing import Literal


def normalize_by_compound(
    df: pd.DataFrame,
    compound: str = "medium",
    agg: Literal["mean", "median"] = "median",
    src_col: str = "value",
    dst_col: str = "value",
    idx=["location", "microscope", "study", "plate", "day"],
) -> pd.DataFrame:
    """Normalize 'value' column by median of compound
    NOTE: rows with "empty" treatment are ignored.

    Args:
        df (pd.DataFrame): Dataframe with sampled values
        compound (str, optional): name of compound to normalize by. Defaults to 'medium'.
        agg (Literal[mean, median], optional): How to normalize, can use anything pd.agg() accepts. Defaults to 'median'.
        col (str): column to put normalized value


    Raises:
        Exception: raised if <remove_no_norm> is False and found samples with nothing to normalize by

    Returns:
        pd.DataFrame: same dataframe but 'value' column is normalized by <compound>
    """
    has_compound = df.groupby(idx).apply(lambda df: any((df.compound == compound)))
    df = df.set_index(idx)
    nothing = ~has_compound
    if nothing.sum() > 0:
        raise Exception(f"Found samples with nothing to norm by {nothing[nothing]}")

    norm = df.groupby(idx).apply(
        lambda df: df[(df.compound == compound)][src_col].agg(agg)
    )
    df.loc[norm.index, "norm"] = norm
    df[dst_col] = df[src_col] / df["norm"]

    df = df.reset_index()
    for idx, group in df.groupby(idx):
        control = group.query(f'(compound == "{compound}")').sort_values(src_col)
        df.loc[group.index, "control"] = [str(control.index.to_list())] * len(group)
    df["control"] = df.control.map(eval)

    return df


def set_control(
    df: pd.DataFrame,
    compound: str = "medium",
    idx=["location", "microscope", "study", "plate", "day"],
):
    df = df.copy()

    has_compound = df.groupby(idx).apply(lambda df: any((df.compound == compound)))
    df = df.set_index(idx)
    nothing = ~has_compound
    if nothing.sum() > 0:
        raise Exception(f"Found samples with no control {nothing[nothing]}")

    df = df.reset_index()
    for idx, group in df.groupby(idx):
        control = group.query(f'(compound == "{compound}")')
        df.loc[group.index, "control"] = [set(control.index.values)] * len(group)
    df["control"] = df.control.map(list)
    return df


class NormByCompound:
    def __init__(self, compound="medium") -> None:
        super().__init__()
        self.compound = compound

    def __call__(self, df):
        return normalize_by_compound(df, self.compound)


class SetControl:
    def __init__(self, compound="medium") -> None:
        super().__init__()
        self.compound = compound

    def __call__(self, df):
        return set_control(df, self.compound)


class SetDomainSplit:
    def __init__(self, domain, split) -> None:
        super().__init__()
        self.domain = domain
        self.split = split

    def __call__(self, df):
        df = df.copy()
        df.loc[df.domain == self.domain, "split"] = self.split
        return df
