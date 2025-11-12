import pandas as pd


def merge_features_with_clinical_data(
    other_tsv: str,
    clinical_tsv: str,
    merge_on: str = "PATIENT_ID",
    separator: str = "\t",
    skiprows_on_clinical: int = 4,
    skiprows_on_other: int = 0,
) -> pd.DataFrame:
    """Merges a selected df from a tsv with clinical data, needs to have a
    column with the same name to merge on.

    Args:
        other_tsv (str): string path to features to merge with clinical data
        clinical_tsv (str): string path to clinical data
        merge_on (str, optional): what to merge on. Defaults to "PATIENT_ID".
        separator (str, optional): Defaults to "\t".
        skiprows_on_clinical (int, optional): Defaults to 4.
        skiprows_on_other (int, optional): Defaults to 0.

    Returns:
        pd.DataFrame: returns the merged df
    """
    data1: pd.DataFrame = pd.read_csv(
        clinical_tsv, sep=separator, skiprows=skiprows_on_clinical
    )
    data2: pd.DataFrame = pd.read_csv(
        other_tsv, sep=separator, skiprows=skiprows_on_other
    )

    if "-" in data1[merge_on] and "_" in data2[merge_on]:
        data1[merge_on] = data1[merge_on].str.replace("-", "_")

    elif "-" in data2[merge_on] and "_" in data1[merge_on]:
        data2[merge_on] = data2[merge_on].str.replace("-", "_")

    merged: pd.DataFrame = data1.merge(data2, how="inner", on=merge_on)
    return merged
