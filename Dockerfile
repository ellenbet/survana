# we base our image on a slim version with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

#
WORKDIR /app

ENV UV_COMPULE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR =/usr/local/bin

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

EXPOSE 8000

CMD ["fastapi", "dev", "--host", "0.0.0.0", "8000"]
