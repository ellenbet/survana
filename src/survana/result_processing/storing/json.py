import json
import os

from survana.config import PATHS


def load_results(path=PATHS["RESULT_JSON_FILE"]):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"runs": {}, "counts": {}}


# TODO - interaction between plotter and results storage instead

"""def update_results(strings, path=PATHS["RESULT_JSON_FILE"]):
    data = load_results(path)

    run_id = datetime.now().isoformat(timespec="seconds")

    data["runs"][run_id] = strings

    run_counts = Counter(strings)
    for s, n in run_counts.items():
        data["counts"][s] = data["counts"].get(s, 0) + n

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data"""
