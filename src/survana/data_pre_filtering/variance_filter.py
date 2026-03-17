import pandas as pd

from survana.config import CONFIG, PATHS


def variance_filter(
    raw_features_pth: str = str(PATHS["RAW_FEATURES_PATH"]),
    sep=CONFIG["columns"]["sep"],
    cutoff: int = 1000,
    na_thresh: float = 0.8,
    save=False,
):

    df: pd.DataFrame = pd.read_csv(raw_features_pth, sep=sep)

    features = df.shape[0]
    patient = df.shape[1]
    thresh = patient * na_thresh
    filtered_df = df.dropna(axis=0, thresh=thresh)

    print("ORIGINAL FEATURE AMOUNT", features)
    print("POST-TRIM FEATURE AMOUNT", len(filtered_df))

    top_variance = filtered_df.iloc[:, 3:].var(axis=1)
    top_features = top_variance.nlargest(cutoff)
    top_feature_df = filtered_df.loc[top_features.index]
    locs = ["chrom", "start", "end"]
    top_feature_df["_".join(locs)] = top_feature_df[locs].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )
    feats = top_feature_df
    feats.set_index(top_feature_df["chrom_start_end"], inplace=True)
    feats.pop("chrom_start_end")
    feats.pop("chrom")
    feats.pop("start")
    feats.pop("end")
    df = feats.T
    df.rename_axis(columns={"chrom_start_end": "patient_id"}, inplace=True)

    if save:
        df.to_csv(
            path_or_buf=PATHS["PREFILTERED_DATA_PATH_VARIANCE"]
            / f"variance_filtered_features_{cutoff}.csv"
        )
    return df


if __name__ == "__main__":
    variance_filter(save=True)
