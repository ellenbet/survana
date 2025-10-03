import optuna

from models.coxph import coxph

optuna.logging.set_verbosity(optuna.logging.ERROR)
coxph()
