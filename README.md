# tfm-alzheimer-classification
This repository contains everything it is needed to execute the code of the "Metabolites selection as predictors of
Alzheimer disease using classification models" UNED TFM.


## Table of Contents

- [Getting started](#getting-started)


## Getting started

This project uses a Docker image to containerize the development environment, Poetry as the Python dependency manager,
and Ruff as the project's linter. Workflows run inside the container so everyone uses the same system packages and
Python version. From within the container you can open the VS Code workspace and install the recommended extensions
(for example: `ms-python.python` and `charliermarsh.ruff`) so Ruff provides linting and formatting inside the editor.

Follow these steps to build the development image, run a container with the repository mounted, and create the project
virtual environment with Poetry.

- Build the Docker image from the repository root:

```bash
docker build -t tfm-alzheimer-utils -f utils/Dockerfile .
```

- Run an interactive container from the directory that contains the repository on it and mount that at `/workspace`:

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  -u "$(id -u):$(id -g)" \
  tfm-alzheimer-utils
```

- Install project dependencies with Poetry:

```bash
poetry install --with=dev
```

- Activate the virtual environment (one of):

```bash
source .venv/bin/activate
# or
poetry shell
```

That's it — you can now develop inside the container with dependencies installed in `./.venv`.
