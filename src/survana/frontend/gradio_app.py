# survana/src/modules/frontend.py

import shutil
import threading
import tomllib
from pathlib import Path
from typing import Any

import gradio as gr
import tomli_w
from PIL import Image

from survana.config import CONFIG, PATHS
from survana.data_processing.dataloaders import load_data_for_sksurv_coxnet
from survana.models.stability_selection import (
    StabilitySelectionCancelled,
    stability_selection,
)
from survana.result_processing.single_stability_result import (
    SingleStabilityResult,
)

BASE_DIR = Path(__file__).parent
IMG_PATH = BASE_DIR / "imgs" / "ChatGPT Image 13. feb. 2026, 17_38_44.png"


img = Image.open(IMG_PATH)
img = img.resize((2000, 400), Image.LANCZOS)  # type: ignore
img.save(IMG_PATH)

CSS = """
/* Tab button centering */
.gradio-container .tabs > div:first-child {
    display: flex !important;
    justify-content: center !important;
}

/* Optional: keep the tab row from stretching weirdly */
.gradio-container .tabs > div:first-child button {
    margin: 0 6px !important;
}
"""

STOP_REQUESTED = threading.Event()
STABILITY_SELECTION_RUNNING = threading.Event()


def request_stability_selection_stop():
    if STABILITY_SELECTION_RUNNING.is_set():
        STOP_REQUESTED.set()
        return (
            "<span style='color:#666;font-size:0.95em;'>"
            "Stopping stability selection...</span>"
        )
    return ""


def load_toml_state(path_str: str):
    path = Path(path_str)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    # NEW: return empty undo history on load
    return data, path_str, [], tomli_w.dumps(data)


def set_toml_value(
    data: dict, section: str, key: str, value: str, value_type: str
):
    """Update TOML dict in memory and return updated dict
    + endered TOML text."""
    if not key:
        return data, tomli_w.dumps(data)

    # Navigate/create section
    target = data
    section = (section or "").strip()
    if section:
        for part in section.split("."):
            target = target.setdefault(part, {})

    # Parse value
    v: Any = value
    if value_type == "int":
        v = int(value)
    elif value_type == "float":
        v = float(value)
    elif value_type == "bool":
        v = value.strip().lower() in {"1", "true", "yes", "y", "on"}
    # else: keep as string

    target[key] = v
    return data, tomli_w.dumps(data)


def apply_change(
    data: dict,
    history_list,
    section: str,
    key: str,
    value: str,
    value_type: str,
):
    """
    Push current config into history, then apply change.
    Returns: updated_data, updated_history, rendered_toml
    """
    history = list(history_list)  # copy
    history.append(tomli_w.dumps(data))  # snapshot BEFORE change

    data, rendered = set_toml_value(data, section, key, value, value_type)
    return data, history, rendered


def apply_change_and_clear(
    data, history_list, section, key, value, value_type
):
    data, history, rendered = apply_change(
        data, history_list, section, key, value, value_type
    )
    return data, history, rendered, "", "", ""  # clear section/key/value


def undo_last(data: dict, history_list):
    """
    Restore previous snapshot if available.
    Returns: restored_data, updated_history, rendered_toml
    """
    history = list(history_list)  # copy
    if not history:
        # nothing to undo
        return data, history, tomli_w.dumps(data)

    prev_text = history.pop()
    prev_data = tomllib.loads(prev_text)
    return prev_data, history, prev_text


def save_toml_to_disk(data: dict, path_str: str):
    path = Path(path_str)
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    return f"✅ Saved to {path}"


def run_stability_selection(
    clinical_data_file: str | None,
    epigenetic_data_file: str | None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    progress(0, desc="Preparing inputs")
    STOP_REQUESTED.clear()
    STABILITY_SELECTION_RUNNING.set()
    loader_kwargs = {}
    if epigenetic_data_file:
        loader_kwargs["path_to_epigenetic_features"] = epigenetic_data_file
    if clinical_data_file:
        loader_kwargs["path_to_clinical_data"] = clinical_data_file

    progress(0.1, desc="Loading clinical and epigenetic data")
    data_collection = load_data_for_sksurv_coxnet(**loader_kwargs)
    progress(0.2, desc="Running stability selection")
    try:
        result = stability_selection(
            data_collection=data_collection,
            stop_requested=STOP_REQUESTED.is_set,
        )
    except StabilitySelectionCancelled:
        STABILITY_SELECTION_RUNNING.clear()
        STOP_REQUESTED.clear()
        return (
            "",
            "<span style='color:#666;font-size:0.95em;'>"
            "Stability selection stopped.</span>",
            gr.update(value=None, visible=False),
        )

    STABILITY_SELECTION_RUNNING.clear()
    STOP_REQUESTED.clear()

    selected_summary = (
        ", ".join(result.get_selected_features()) or "No features selected."
    )
    progress(0.95, desc="Formatting results")
    progress(1, desc="Done")
    return (
        selected_summary,
        "",
        gr.update(
            value=result.get_stability_path_with_thresh_figure(),
            visible=True,
        ),
    )


def load_stability_selection_result(result_file: str | None):
    if not result_file:
        return (
            "Please upload a result CSV file.",
            "",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )

    try:
        normalized_result_path = _prepare_uploaded_result_file(
            Path(result_file)
        )
        result = SingleStabilityResult(normalized_result_path)
        selected_summary = (
            ", ".join(result.get_selected_features())
            or "No features selected."
        )
        return (
            "",
            selected_summary,
            gr.update(
                value=result._build_stability_path_figure(), visible=True
            ),
            gr.update(
                value=result._build_stability_path_with_thresh_figure(),
                visible=True,
            ),
        )
    except Exception as exc:
        return (
            f"Could not load result file: {exc}",
            "",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )


def _prepare_uploaded_result_file(result_file: Path) -> Path:
    expected_dir = PATHS["RESULT_CSV_DATA_PATH"]
    expected_dir.mkdir(parents=True, exist_ok=True)

    if (
        result_file.parent == expected_dir
        and "log(lambda)_" in result_file.name
        and "_to_" in result_file.name
        and "_results" in result_file.name
    ):
        return result_file

    min_lambda = CONFIG["tuning"]["log_lambda_min"]
    max_lambda = CONFIG["tuning"]["log_lambda_max"]
    normalized_name = (
        f"log(lambda)_{min_lambda}_to_{max_lambda}_results_"
        f"{result_file.stem}.csv"
    )
    normalized_path = expected_dir / normalized_name
    shutil.copy2(result_file, normalized_path)
    return normalized_path


def build_app():
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("# Survana Demo")
        gr.Image(
            value=str(IMG_PATH),
            show_label=False,
            interactive=False,
            height=400,
            width=2000,
        )

        with gr.Tabs():
            with gr.Tab("Stability selection"):
                gr.Markdown("## Run stability selection")
                gr.Markdown(
                    "Upload clinical and epigenetic data files, or leave them "
                    "empty to use the paths from `config.toml`."
                )

                clinical_data_file = gr.File(
                    label="Clinical data file",
                    type="filepath",
                    file_types=[".csv", ".txt", ".tsv"],
                )
                epigenetic_data_file = gr.File(
                    label="Epigenetic data file",
                    type="filepath",
                    file_types=[".csv", ".txt", ".tsv"],
                )
                with gr.Row():
                    go = gr.Button("Run stability selection")
                    stop = gr.Button("Stop", variant="stop")
                selected_features = gr.Textbox(
                    label="Selected features",
                    interactive=False,
                    lines=6,
                )
                stability_path_thresh_plot = gr.Plot(
                    label="Stability path with threshold",
                    visible=False,
                )
                stop_warning = gr.Markdown("")

                go.click(
                    fn=run_stability_selection,
                    inputs=[
                        clinical_data_file,
                        epigenetic_data_file,
                    ],
                    outputs=[
                        selected_features,
                        stop_warning,
                        stability_path_thresh_plot,
                    ],
                    show_progress="minimal",
                    show_progress_on=selected_features,
                )
                stop.click(
                    fn=request_stability_selection_stop,
                    outputs=stop_warning,
                    queue=False,
                )

            with gr.Tab("Display results from stability selection"):
                gr.Markdown("## Display results from stability selection")
                gr.Markdown(
                    "Upload a result CSV with the same format as "
                    "`gradio_result_csv_test/log(lambda)...` to visualize "
                    "the stability paths."
                )

                result_csv_file = gr.File(
                    label="Stability selection result file",
                    type="filepath",
                    file_types=[".csv"],
                )
                load_result_button = gr.Button("Load result")
                load_result_status = gr.Markdown("")
                loaded_selected_features = gr.Textbox(
                    label="Selected features",
                    interactive=False,
                    lines=6,
                )
                stability_path_plot = gr.Plot(
                    label="Stability path",
                    visible=False,
                )
                stability_path_thresh_result_plot = gr.Plot(
                    label="Stability path with threshold",
                    visible=False,
                )

                load_result_button.click(
                    fn=load_stability_selection_result,
                    inputs=result_csv_file,
                    outputs=[
                        load_result_status,
                        loaded_selected_features,
                        stability_path_plot,
                        stability_path_thresh_result_plot,
                    ],
                )

            with gr.Tab("Update and inspect model settings") as config_tab:
                # ---- Top row: path + load ----
                with gr.Row():
                    path_inp = gr.Textbox(
                        value="config.toml", label="Path", visible=False
                    )

                # State: toml dict + current path + undo history
                toml_state = gr.State({})
                current_path = gr.State("config.toml")
                toml_history = gr.State([])  # NEW

                # ---- Two columns: editor (left) + preview (right) ----
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Add or edit a field")

                        section = gr.Textbox(
                            label="Section (optional)",
                            placeholder="e.g paths, model",
                        )
                        key = gr.Textbox(
                            label="Key",
                            placeholder="e.g. base_dir, model_type",
                        )
                        value = gr.Textbox(
                            label="Value", placeholder="e.g. lasso, 100, 0.1"
                        )
                        value_type = gr.Radio(
                            ["str", "int", "float", "bool"],
                            value="str",
                            label="Type",
                        )

                        apply_btn = gr.Button("Add")
                        undo_btn = gr.Button("Undo")  # NEW
                        save_btn = gr.Button("Save to file")

                        status = gr.Markdown("")

                    with gr.Column(scale=3):
                        gr.Markdown("config.toml")
                        toml_view = gr.Code(
                            label="Live TOML preview", language="markdown"
                        )

                # Auto-load whenever you enter this tab
                config_tab.select(
                    fn=load_toml_state,
                    inputs=path_inp,
                    outputs=[
                        toml_state,
                        current_path,
                        toml_history,
                        toml_view,
                    ],
                )

                # Apply -> push previous state into history + update preview
                apply_btn.click(
                    fn=apply_change_and_clear,
                    inputs=[
                        toml_state,
                        toml_history,
                        section,
                        key,
                        value,
                        value_type,
                    ],
                    outputs=[
                        toml_state,
                        toml_history,
                        toml_view,
                        section,
                        key,
                        value,
                    ],
                )

                # Undo -> restore last snapshot (if any)
                undo_btn.click(
                    fn=undo_last,
                    inputs=[toml_state, toml_history],
                    outputs=[toml_state, toml_history, toml_view],
                )

                # Save button writes current state to disk
                save_btn.click(
                    fn=save_toml_to_disk,
                    inputs=[toml_state, current_path],
                    outputs=status,
                )

    return demo.queue()


if __name__ == "__main__":
    app = build_app()
    app.launch()
