FROM ubuntu:24.04

ARG DEFAULT_USER=ubuntu
ARG POETRY_VERSION=2.2.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3.12 \
    python3.12-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Align python3 to Python 3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Create a non-root user and set permissions for the workspace
RUN mkdir -p /workspace \
    && chown -R "$DEFAULT_USER":"$DEFAULT_USER" /workspace

# Explicitly set the default user
USER "$DEFAULT_USER"
WORKDIR /workspace

# Make Poetry binaries and settings available at runtime
ENV PATH="/home/${DEFAULT_USER}/.local/bin:${PATH}"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_VIRTUALENVS_PATH="./.venv"

# Install Poetry as the default user
RUN curl -sSL https://install.python-poetry.org | python3 - --version "$POETRY_VERSION" \
    && mkdir -p "$HOME/.local/bin" \
    && export PATH="$HOME/.local/bin:$PATH" \
    && printf '\nexport PATH="$HOME/.local/bin:$PATH"\n' >>/home/$DEFAULT_USER/.bashrc \
    && printf '\nexport POETRY_VIRTUALENVS_IN_PROJECT=true' >>/home/$DEFAULT_USER/.bashrc \
    && printf '\nexport POETRY_VIRTUALENVS_PATH="./.venv"\n' >>/home/$DEFAULT_USER/.bashrc \
    && "$HOME"/.local/bin/poetry self add "poetry-dynamic-versioning[plugin]" \
    && "$HOME"/.local/bin/poetry self add "poetry-plugin-shell"

# Switch to root to create symbolic link for global access
USER root
RUN ln -sf "/home/$DEFAULT_USER/.local/bin/poetry" /usr/local/bin/poetry \
    && printf '\nexport PATH="/home/$DEFAULT_USER/.local/bin:$PATH"\n' >>/etc/bash.bashrc

# Switch back to default user
USER "$DEFAULT_USER"

CMD ["bash"]
