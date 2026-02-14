# survana/src/modules/frontend.py

import tomllib
from pathlib import Path
from typing import Any

import gradio as gr
import tomli_w
from PIL import Image

from survana.data_pre_filtering.domain_filter import domain_filter

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


def run_domain_filter():
    return domain_filter().iloc[:100, :3]


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
            with gr.Tab("Domain filter"):
                gr.Markdown("## Demo for domain filter")

                go = gr.Button("Run domain filter")
                result = gr.Dataframe(
                    label="Preview of domain-specific features:",
                    interactive=False,
                    wrap=True,
                    row_count=3,
                    col_count=(None, "fixed"),
                )

                go.click(
                    fn=run_domain_filter,
                    outputs=result,
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

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
