from atp.dili.ic50 import calc_ic50, calc_ic_frac
import pandas as pd
import numpy as np
from atp.preprocess import normalize_by_compound

unlabeled = pd.read_pickle('./data/data_thorlabs_all_labeled.pickle')
unlabeled = unlabeled[unlabeled.value.notna()]

# filter out drugs not in proctor
def main(unlabeled: pd.DataFrame, normalize_by_dmso: bool, add_conc_0_1: bool, pl:int, set_inf_extrapolating:bool):
    proctor_df = pd.read_csv('./data/proctor.csv')
    compounds = proctor_df.compound.unique().tolist() + ['dmso']
    unlabeled = unlabeled[unlabeled.eval('compound in @compounds')]

    if normalize_by_dmso:
        has_dmso = unlabeled.groupby(['plate', 'day']).apply(lambda x : 'dmso' in x.compound.tolist())
        can_norm = unlabeled.set_index(['plate', 'day']).loc[has_dmso[has_dmso].index]
        # cant_norm = unlabeled.set_index(['plate', 'day']).loc[has_dmso[~has_dmso].index]

        can_norm = normalize_by_compound(can_norm.reset_index(), src_col='value', dst_col='value', compound='dmso')
        normalized_without_plate = normalize_by_compound(unlabeled, src_col='value', dst_col='value', compound='dmso', idx=["location", "microscope", "study", "day"])

        normalized = can_norm.location_in_bucket.tolist()
        normalized_without_plate = normalized_without_plate[normalized_without_plate.eval('location_in_bucket not in @normalized')]
        unlabeled = pd.concat([can_norm, normalized_without_plate], ignore_index=True)

    unlabeled = unlabeled.query('compound != "dmso"')
    for_ic50 = unlabeled.groupby(['day', 'compound', 'concentration_uM'])[['value']].median().reset_index()

    if add_conc_0_1:
        f = lambda group: calc_ic50([0] + group.concentration_uM.tolist(), [1] + group.value.tolist(), pl)
    else:
        lambda group: calc_ic50(group.concentration_uM.tolist(), group.value.tolist(), pl)
    ic50 = for_ic50.groupby(['day', 'compound']).apply(f)
    valid, ic50 = ic50.map(lambda x: x[0]), ic50.map(lambda x: x[1])
    ic50_orig = pd.DataFrame(ic50.map(pd.Series).tolist(), index=ic50.index)

    ic50_orig['concentrations'] = for_ic50.groupby(['day', 'compound']).apply(lambda x: x.concentration_uM.values)
    ic50_orig['values'] = for_ic50.groupby(['day', 'compound']).apply(lambda x: x.value.values)

    ic50 = ic50_orig.copy()

    failed_fit_idx = ic50.ic50.isna()
    failed_fit = ic50[failed_fit_idx]
    # ic50[~failed_fit_idx]
    print('failed fit - ', len(failed_fit))

    rejected_upper_idx = (ic50.upper_plato < 0)
    rejected_upper = ic50[rejected_upper_idx]
    # ic50[~rejected_upper_idx]
    print('rejected upper -', len(rejected_upper))

    rejected_lower_idx = (ic50.lower_plato > 1.3) | (ic50.lower_plato < -2)
    rejected_lower = ic50[rejected_lower_idx]
    # ic50[~rejected_lower_idx]
    print('rejected lower -', len(rejected_lower))

    upper_lower_idx = (ic50.lower_plato > ic50.upper_plato)
    upper_lower = ic50[upper_lower_idx]
    # ic50[~upper_lower_idx]
    print('upper < lower -', len(upper_lower))

    inf_ic50_idx = ic50.lower_plato > 0.5
    inf_ic50 = ic50[inf_ic50_idx].copy()
    ic50['ic50_inf'] = ic50.ic50
    ic50.loc[inf_ic50_idx, 'ic50_inf'] = np.inf
    inf_ic50['ic50'] = np.inf
    print('set inf ic50 -', len(inf_ic50))

    for ic in [10, 15, 20, 25, 30, 40, 50]:
        if pl == 4:
            ic50[f'ic{ic}_calc'] = ic50.apply(lambda row: calc_ic_frac(row.upper_plato, row.hill_coeff, row.ic50, row.lower_plato, frac=ic / 100, pl=pl), axis=1)
        if pl == 2:
            ic50[f'ic{ic}_calc'] = ic50.apply(lambda row: calc_ic_frac(row.hill_coeff, row.ic50, frac=ic / 100, pl=pl), axis=1)
        if set_inf_extrapolating:
            ic50[f'ic{ic}_calc'] = ic50.apply(lambda row: np.inf if max(row.concentrations * 1.5) < (row[f'ic{ic}_calc']) else row[f'ic{ic}_calc'], axis=1)


    ic50 = ic50.reset_index().merge(proctor_df[['compound', 'cmax', 'binary_label']], on='compound', how='left')
    return ic50

ic50 = main(unlabeled, normalize_by_dmso=True, add_conc_0_1=True, pl=2, set_inf_extrapolating=True)
ic50.to_pickle('./data/ic50_ctg.pickle')

print(ic50)