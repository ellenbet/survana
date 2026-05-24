# Survival analysis and Biomarker discovery

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/0ed3e16c-3415-49f0-b339-d4de0ca665a8" />

Survival analysis is a branch of statistics applied to modeling time-to-event data. It is used in time prediction from diagnosis until event, where an event can be considered either death or reoccurrence. In other fields, this topic is known as reliability analysis (engineering), duration analysis (economics), and event history analysis (sociology).


### 1. Explore stability in feature selection by artificial feature injection
Inspired by the Stabl repository and similarly named Nature publication, we inject artificial features into the design matrix for later feature selection, which allows for FDR-proxy error control with the ultimate goal of selecting a cut-off on the minimum allowed feature frequency during subsampled feature selection runs.

# How to use:
Note that this is a very unfinished repository and it is not yet suitable to be used as a Python package.

## Use package:
Feel free to use the repo as is by cloning it, install package after cloning using uv command:

``uv pip install -e .``

## Read documentation:
Until the documentation is added to Github, it can be built and served through the mkdocs package. To build the documentation, use the following command:

``mkdocs build``

To serve the documentation on your localhost, use the following command:

``mkdocs serve``

Note that this is still a work in progress.

## Tests:
When making changes to repository, test continiously using command:

``pytest tests``

To include full integration test, run command:

``pytest tests --slow``

## Gradio:
To run hot reloading gradio for front-end developement, use:

``gradio src/survana/frontend/gradio_app.py``
