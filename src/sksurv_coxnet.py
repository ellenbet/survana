import logging
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sksurv.linear_model as lm
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from config import PREFILTERED_DATA_PATH
from dataloaders import load_data_for_sksurv_coxnet

SEED: int = 42
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger("CoxNet")


CENSOR_STATUS: str = "RFS_STATUS"
MONTHS_BEFORE_EVENT: str = "RFS_MONTHS"


data_collection: tuple[
    pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[Any]]
] = load_data_for_sksurv_coxnet(
    str(PREFILTERED_DATA_PATH),
    response_variables=(CENSOR_STATUS, MONTHS_BEFORE_EVENT),
)


data: pd.DataFrame = data_collection[0]  # Full dataframe
X: pd.DataFrame = data_collection[1]  # Just designmatrix
# Response variable, array with dim n (patients) x 2
y: np.recarray[tuple[Any, ...], np.dtype[Any]] = data_collection[2]


logger.info(
    "\nCensored patients in data:",
    len(data[CENSOR_STATUS]) - data[CENSOR_STATUS].sum(),
    "out of",
    len(data[CENSOR_STATUS]),
)


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)


# setting up alphas - doesnt work if alpha < 10ˆ-3
alphas: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = np.logspace(
    -2, 0, 10
)
estimator = lm.CoxnetSurvivalAnalysis(
    l1_ratio=1,
    fit_baseline_model=True,
    verbose=True,
    alphas=alphas,
    max_iter=100000,
)


logger.info(
    "\nFitting Cox Net model to our data for"
    + f"{len(estimator.alphas)} alphas..."
)
estimator.fit(x_train, y_train)
# .alphas_ are the actual parameters left after the fitting process
print(f"\nSuccess in fitting a total of {len(estimator.alphas_)} alphas...")


coxnet_pipe: Pipeline = make_pipeline(
    lm.CoxnetSurvivalAnalysis(
        l1_ratio=1.0,
        fit_baseline_model=True,
        verbose=True,
        alphas=alphas,
        max_iter=1000000,
    )
)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FitFailedWarning)


# TODO change this to train
coxnet_pipe.fit(x_train, y_train)
estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
cv = KFold(n_splits=5, shuffle=True, random_state=0)
gcv = GridSearchCV(
    make_pipeline(lm.CoxnetSurvivalAnalysis(l1_ratio=1)),
    param_grid={
        "coxnetsurvivalanalysis__alphas": [
            [v] for v in map(float, estimated_alphas)
        ]
    },
    cv=cv,
    error_score=0.5,
    n_jobs=-1,
).fit(x_train, y_train)


"""
Concordance index results
"""
cv_results = pd.DataFrame(gcv.cv_results_)
alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
mean = cv_results.mean_test_score
std = cv_results.std_test_score

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(alphas, mean)
ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
ax.set_xscale("log")
ax.set_ylabel("c-index")
ax.set_xlabel(r"$\lambda$")
ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
ax.axhline(0.5, color="grey", linestyle="--")
ax.grid(True)
plt.savefig("c_index.pdf")
# plt.show()

"""
Best estimators
"""
# TODO one model for the best lambda! -> done below

best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
best_alpha = np.zeros(1)
best_alpha[0] = gcv.best_params_["coxnetsurvivalanalysis__alphas"][0]
print("\n\nbest lambda:", best_alpha[0])
estimator = lm.CoxnetSurvivalAnalysis(
    l1_ratio=1,
    fit_baseline_model=True,
    verbose=False,
    alphas=best_alpha,
    max_iter=100000,
)
estimator.fit(x_train, y_train)
risk_scores = x_test @ estimator.coef_
print("estimator c-index:", estimator.score(x_test, y_test))


# best_coefs = pd.DataFrame(best_model.coef_,
# index = X.columns, columns=["coefficient"])
best_coefs = pd.DataFrame(
    estimator.coef_, index=X.columns, columns=["coefficient"]
)
non_zero = np.sum(best_coefs.iloc[:, 0] > 10e-4)


print(f"Number of non-zero coefficients: {non_zero}")
non_zero_coefs = best_coefs.query("abs(coefficient) > 10e-4")
coef_order = non_zero_coefs.abs().sort_values("coefficient").index

_, ax = plt.subplots(figsize=(7, 5))
non_zero_coefs.loc[coef_order][:20].plot.barh(ax=ax, legend=False)
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("feature_id")
ax.grid(True)
plt.tight_layout()
# plt.title("Non-zero coefficients based on importance")
plt.savefig("best_coefs.pdf")
# plt.show()


"""
Kaplan Meyer plot based on best coefficients
(selected by top concordance index)

"""


def merge_data(other_tsv, clinical_tsv, merge_on="PATIENT_ID"):
    data1 = pd.read_csv(
        clinical_tsv, sep="\t", skiprows=4
    )  # the clinical file has 5 rows with non-df data
    data1["PATIENT_ID"] = data1["PATIENT_ID"].str.replace(
        "-", "_"
    )  # the clinical file has - instead of _ to separate patient ids
    data2 = pd.read_csv(other_tsv, sep="\t")
    merged = data1.merge(data2, how="inner", on=merge_on)
    return merged


merged_df = merge_data(
    "../data/stand_knn_features_variance1000.tsv",
    "../data/data_clinical_patient.txt",
)
X["PATIENT_ID"] = data["PATIENT_ID"]
assert (
    X["PATIENT_ID"].all() == merged_df["PATIENT_ID"].all()
), "not able to sort X by merged due to mismatch of patient id order"

"""
Option to select a filter found in clinical data,
such as claudin subtype (LumA, LumB, etc..)


# decide on filtering factor:
filter = "CLAUDIN_SUBTYPE"
assert filter in merged_df.columns, "WARNING: not found in merged df columns"

# encoding legend handles to get static color
X[filter] = merged_df[filter]
X.sort_values(by = filter, inplace = True)
subtypes = set(X[filter])
subtypes = list(subtypes)
encoder = {}
for ind in range(len(subtypes)):
    encoder[subtypes[ind]] = ind


Fitting tbased on the best params

Is this actually not a bit weird because we have all of our coefficients,
perhaps better to just fit based on those?



coxnet_pred = make_pipeline(lm.CoxnetSurvivalAnalysis(l1_ratio=0.99,
fit_baseline_model=True))
coxnet_pred.set_params(**gcv.best_params_)
fit_X = X.drop(["PATIENT_ID", filter], axis = 1)
coxnet_pred.fit(fit_X , y)
surv_fns = coxnet_pred.predict_survival_function(fit_X)


Plotting



time_points = np.quantile(y["RFS_MONTHS"], np.linspace(0, 0.6, 100))
legend_handles = []
legend_labels = []
_, ax = plt.subplots(figsize=(9, 6))

for fn, label in zip(surv_fns, X.loc[:, filter].astype(str)):
    (line,) = ax.step(time_points, fn(time_points), where="post",
    color=f"C{encoder[label]}", alpha=0.5)
    if label not in legend_labels:
        legend_labels.append(label)
        legend_handles.append(line)

ax.legend(legend_handles, legend_labels)
ax.set_xlabel("time")
ax.set_ylabel("Survival probability")
ax.grid(True)
plt.title(f"Kaplan Meier with best coefficients, filtered by {filter}")
#plt.show()
"""


merged_df["risk_score"] = X.drop(["PATIENT_ID"], axis=1) @ estimator.coef_
merged_df["binary_risk"] = (
    merged_df["risk_score"] > np.median(merged_df["risk_score"])
).astype(int)

"""
# decide on filtering factor:
filter = "binary_risk"
assert (
    filter in merged_df.columns
), "WARNING: suggested filter not found in merged df columns"

# encoding legend handles to get static color
X[filter] = merged_df[filter]
X.sort_values(by=filter, inplace=True)
subtypes: set[Any] = set(X[filter])
subtypes = [str(subtype) for subtype in subtypes]
encoder = {}
for ind in range(len(subtypes)):
    encoder[subtypes[ind]] = ind


Fitting tbased on the best params

Is this actually not a bit weird because we have all of our coefficients,
perhaps better to just fit based on those?



coxnet_pred = make_pipeline(
    lm.CoxnetSurvivalAnalysis(l1_ratio=1, fit_baseline_model=True)
)
coxnet_pred.set_params(**gcv.best_params_)
fit_X = X.drop(["PATIENT_ID", filter], axis=1)
coxnet_pred.fit(fit_X, y)
surv_fns = coxnet_pred.predict_survival_function(fit_X)



time_points = np.quantile(y["RFS_MONTHS"], np.linspace(0, 0.6, 100))
legend_handles = []
legend_labels = []
_, ax = plt.subplots(figsize=(9, 6))

for fn, label in zip(surv_fns, X.loc[:, filter].astype(str)):
    (line,) = ax.step(
        time_points,
        fn(time_points),
        where="post",
        color=f"C{encoder[label]}",
        alpha=0.5,
    )
    if label not in legend_labels:
        legend_labels.append(label)
        legend_handles.append(line)

ax.legend(legend_handles, legend_labels)
ax.set_xlabel("time (months)")
ax.set_ylabel("Survival probability")
ax.grid(True)
plt.title(f"filtered by {filter}")
plt.show()
"""
