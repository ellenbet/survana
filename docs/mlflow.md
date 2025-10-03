https://medium.com/@egorborisov/introduction-to-mlflow-61f3572ac204

start mlflow on localhost:5000 with ``mlflow ui``

Commonly used methods include:
- ``mlflow.log_params``
    - Logs the hyperparameters

- ``mlflow.log_metric``
    - Logs the metrics

Automatically stores runs in the ./mlruns directory, can change trackig URI by specifying MLFLOW_TRACKING_URI.
