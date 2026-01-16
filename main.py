# optuna.logging.set_verbosity(optuna.logging.ERROR)
import warnings

from survana.models.stability_selection import stability_selection

warnings.filterwarnings("ignore", category=RuntimeWarning)
stability_selection()
