import numpy as np
import pandas as pd


def split_loc_batch_chrono(df, seed, train_frac=0.6, val_frac=0.2):
    locations = df["domain"].unique()
    rng = np.random.default_rng(seed=seed)
    train_index, val_index, test_index = [], [], []
    for c in locations:
        batches = (
            df.query(f"domain == @c")[["batch", "date"]]
            .drop_duplicates()
            .sort_values(["date"])["batch"]
            .to_numpy()
        )
        l = len(batches)
        train_val, test = np.split(batches, [int((train_frac + val_frac) * l)])
        rng.shuffle(train_val)
        train, val = np.split(batches, [int(train_frac * l)])
        train_index += df[df.eval(f"(domain == @c) & (batch in @train)")].index.tolist()
        val_index += df[df.eval(f"(domain == @c) & (batch in @val)")].index.tolist()
        test_index += df[df.eval(f"(domain == @c) & (batch in @test)")].index.tolist()
    return pd.Index(train_index), pd.Index(val_index), pd.Index(test_index)


class ChronologicalLocationBatchSplitter:
    def __init__(self, train_domain, seed) -> None:
        self.train_domain = train_domain
        self.seed = seed
        if train_domain is None:
            self.test_domain = None
        else:
            self.test_domain = (
                "A_echo_rebel"
                if train_domain == "B_echo_revolve_v0"
                else "B_echo_revolve_v0"
            )
        self.train = 0.8
        self.val = 0.2

    def __call__(self, df):
        df = df.copy()

        train_val = df.query('split == "train_val"')
        if self.train_domain != None:
            train_val = train_val.query(f'domain == "{self.train_domain}"')
        train_idx, val_idx, _ = split_loc_batch_chrono(
            train_val, self.seed, self.train, self.val
        )

        if self.test_domain:
            test_idx = df.query(f'domain == "{self.test_domain}"').index
            df.loc[test_idx, "split"] = "test"

        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"

        return df


class RandomSplitter:
    def __init__(self, seed, train_ratio=0.75) -> None:
        self.seed = seed
        self.train_ratio = train_ratio

    def __call__(self, df):
        df = df.copy()

        rng = np.random.default_rng(seed=self.seed)

        train_val = df.query('split == "train_val"')
        idxes = train_val.index.to_numpy()
        rng.shuffle(idxes)
        train_idx, val_idx = np.split(idxes, [int(self.train_ratio * len(train_val))])

        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"

        return df
