# Survival analysis and Biomarker discovery

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/0ed3e16c-3415-49f0-b339-d4de0ca665a8" />

Survival analysis is a branch of statistics applied to modeling time-to-event data. It is used in time prediction from diagnosis until event, where an event can be considered either death or reoccurrence. In other fields, this topic is known as reliability analysis (engineering), duration analysis (economics), and event history analysis (sociology).

This repository has two main goals:

### 1. Explore best practice principles in MLOps using Optuna and MLFlow
So far, simple survival models such as Cox-Lasso from Scikit-Survival used in combination with Optuna for hyperparameter tuning, standard CV techniques and MLflow for monitoring and logging of results have yielded model scores over 10% higher than similar runs outside of this setup.

### 2. Explore stability in feature selection by artificial feature injection and quantification of results
Inspired by the Stabl repository and similarly named Nature publication, we inject artificial features into the design matrix for later feature selection, which allows for FDR-proxy quantification with the ultimate goal of selecting a cut-off on the minimum allowed feature frequency during subsampled feature selection runs.
