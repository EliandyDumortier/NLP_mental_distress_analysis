🛠️ Poetry Environment Setup

This project uses Poetry for dependency management and virtual environments.

    📦 Requirements

    Python 3.12

    Poetry (version ≥ 1.5)

✅ 1. Create the virtual environment and install dependencies

If not already done:

    poetry install

This will automatically create a virtual environment (usually located under .cache/pypoetry/virtualenvs/...).
✅ 2. Activate the environment manually

If you’re using Poetry ≥ 2.0, the poetry shell command might not be available by default.

You can activate the virtual environment directly like this:

    source $(poetry env info --path)/bin/activate

Or, if you want to be explicit, use the full path:

    source /home/utilisateur/.cache/pypoetry/virtualenvs/nlp-tools-DOy_70HC-py3.12/bin/activate

💡 Tip: You can find the exact path using poetry env info --path.

🧼 3. (Optional) Activate virtualenv inside the project folder

To keep the environment local to the project directory:

    poetry config virtualenvs.in-project true
    poetry install

This will create a .venv/ folder directly in your project.