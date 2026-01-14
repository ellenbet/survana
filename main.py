# optuna.logging.set_verbosity(optuna.logging.ERROR)
import warnings

from survana.models.stability_selection import subsampled_stability_coxph

warnings.filterwarnings("ignore", category=RuntimeWarning)
subsampled_stability_coxph()
