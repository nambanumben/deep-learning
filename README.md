# Deep Learning Project

This project is a starting point for deep learning experiments and development. It is organized for clarity and ease of use, with a focus on reproducibility and modularity.

## Quick Start

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd deep_learning
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies with Poetry:
   ```bash
   poetry install
   ```
4. Launch Jupyter Lab or Notebook:
   ```bash
   poetry run jupyter lab
   # or
   poetry run jupyter notebook
   ```
5. Open any notebook in the `src/` or `src/presentations/` folders to get started.

## Poetry Dependency Installation

To install all dependencies for this project using [Poetry](https://python-poetry.org/), follow these steps:

1. **Install Poetry** (if you don't have it):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   # or, with pipx
   pipx install poetry
   ```
   For more details, see the [Poetry installation docs](https://python-poetry.org/docs/#installation).

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   This will create a virtual environment (if one does not exist) and install all dependencies as specified in `pyproject.toml` and `poetry.lock`.

3. **Activate the virtual environment** (optional, for direct shell access):
   ```bash
   poetry shell
   ```
   Or, run commands with `poetry run <command>` (e.g., `poetry run python script.py`).

4. **Updating dependencies**:
   To update all dependencies to the latest allowed versions:
   ```bash
   poetry update
   ```

## VS Code Support

This project includes configuration for Visual Studio Code to enhance your development experience:

- **.vscode/settings.json**: Sets up Python interpreter, Jupyter notebook root, and workspace variables.
- **.vscode/extensions.json**: Recommends useful extensions for Python and Jupyter development.

### Recommended Extensions
- Python (ms-python.python)
- Jupyter (ms-toolsai.jupyter)
- Jupyter Slideshow (ms-toolsai.vscode-jupyter-slideshow)
- Jupyter Keymap (ms-toolsai.jupyter-keymap)
- Jupyter Renderers (ms-toolsai.jupyter-renderers)

### VS Code Variables Used
- `${workspaceFolder}`: Refers to the root of your project.

These settings help ensure a smooth experience with Python, Jupyter notebooks, and environment management in VS Code.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE)
