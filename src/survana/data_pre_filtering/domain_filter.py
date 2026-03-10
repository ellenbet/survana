import numpy as np
import pandas as pd

from survana.config import CONFIG, PATHS


def domain_filter(
    raw_features_pth: str = str(PATHS["RAW_FEATURES_PATH"]),
    sep=CONFIG["columns"]["sep"],
    chrom_path: str = str(
        PATHS["DATA_PATH"] / CONFIG["filter_filenames"]["chrom_hmm"]
    ),
    unibind_path: str = str(
        PATHS["DATA_PATH"] / CONFIG["filter_filenames"]["unibind"]
    ),
    save=False,
):

    assert (
        ".bed" in CONFIG["filter_filenames"]["chrom_hmm"]
    ), ".bed file required for domain filtering"

    assert (
        ".bed" in CONFIG["filter_filenames"]["unibind"]
    ), ".bed file required for domain filtering"

    chrom_df: pd.DataFrame = pd.read_csv(chrom_path, sep=" ", header=None)
    unibind_df: pd.DataFrame = pd.read_csv(unibind_path, sep=" ", header=None)

    selected_enhancer_areas = chrom_df[
        chrom_df[3].isin(["Enhancer", "CTCF+Enhancer"])
    ][0]

    selected_TF_areas = unibind_df[unibind_df[1].isin(["ESR1", "FOXA1"])][0]

    intersection = pd.Series(
        np.intersect1d(selected_enhancer_areas, selected_TF_areas)
    )

    raw_features: pd.DataFrame = pd.read_csv(raw_features_pth, sep=sep)

    locs = ["chrom", "start", "end"]
    for loc in locs:
        assert (
            loc in raw_features.columns
        ), f"{loc} not in raw feature columns, unable to filter"

    raw_features["_".join(locs)] = raw_features[locs].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )
    feats = raw_features
    feats.set_index(feats["chrom_start_end"], inplace=True)
    feats.pop("chrom_start_end")
    feats.pop("chrom")
    feats.pop("start")
    feats.pop("end")
    domain_filtered_features = feats.loc[intersection]
    df = domain_filtered_features.T
    df.rename_axis(columns={"chrom_start_end": "patient_id"}, inplace=True)

    if save:
        df.to_csv(PATHS["DATA_PATH"] / "domain_filtered_features.csv")
    return df


if __name__ == "__main__":
    domain_filter(save=True)
