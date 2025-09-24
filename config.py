import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Displays info messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Logger - logging.getLogger(__name__) reflects the module
logger: logging.Logger = logging.getLogger("config")

BASE_DIR: Path = Path(os.getenv("BASE_DIR", "survana"))
DATA_PATH: Path = BASE_DIR / "data"
SRC_PATH: Path = BASE_DIR / "src"

paths: list[Path] = [BASE_DIR, DATA_PATH, SRC_PATH]

# Printing config overview
logging.info(
    "\nDEFINED PROJECT PATHS:\n" + "\n".join([str(path) for path in paths])
)
