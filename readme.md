# FriendBook 🩺

## 🧐 Context
The "FriendBook" project aims to automatically analyze sentences to detect potential risks of mental disorders (distress) from textual content. It is based on a natural language processing (NLP) model specialized in the analysis of sentiments and implicit emotional emotions. To ensure a detailed understanding of human language and sensitive psychological contexts, the model was developed by fine-tuning (targeted retraining) a well-known pre-trained model: RoBERTa. Thus adapted to the task, the model learns to classify sentences according to the degree of distress severity (no distress, mild, moderate, severe), providing a reliable technological basis for the automated detection of warning signals in the field of mental health.


## 🔍️ Technical Functionnalities
- Visual interface (frontend) with Streamlit,
- NLP model for sentiment analysis,
- Support for English sentences,
- Depression detection based on severity (no distress, mild, moderate, severe).

## 🏗️ Project architecture
```
NLP_mental_distress_analysis/
        |--App/
                |--controller/
                        |--init.py
                        |--test_model.py
                |--data/
                        |--emotions.txt
                |--model/
                        |--results/
                                |--confusion_matrix.png
                                |--metrics_summary.html
                                |--training_history.html
                        |--init.py
                        |--fine_tune_distress_model.py
                        |--fine_tune_google_collab.py
                        |--go_emotions.py
                        |--training_results.json
                |--Streamlit/
                        |--assets/
                                |--about_enhanced.html
                        |--results/
                        |--static/
                                |--custom.css
                        |--app.py
                        |--emotion_chart.py
                |--__init__.py
        |--.gitattributes
        |--.gitignore
        |--poetry.lock
        |--pyproject.toml
        |--readme.md
```

## 🚀 Lauch the project quickly

#### 0. 📦 Requirements
    Python 3.12
    Poetry (version ≥ 1.5)

#### 1. 👥 Clone the project
```bash
git clone https://github.com/EliandyDumortier/NLP_mental_distress_analysis.git
cd NLP_mental_distress_analysis
```

#### 2. ✅ Create the virtual environment and install dependencies
```bash
git poetry install
```
This will automatically create a virtual environment (usually located under .cache/pypoetry/virtualenvs/...).


#### 3. ✅ Activate the environment manually

If you’re using Poetry ≥ 2.0, the poetry shell command might not be available by default.
You can activate the virtual environment directly like this:

``` bash
source $(poetry env info --path)/bin/activate
```
Or, if you want to be explicit, use the full path:
```bash
source /home/utilisateur/.cache/pypoetry/virtualenvs/nlp-tools-DOy_70HC-py3.12/bin/activate
```
💡 Tip: You can find the exact path using poetry env info --path.

#### 4. 🧼(Optional) Activate virtualenv inside the project folder

To keep the environment local to the project directory:
``` bash
    poetry config virtualenvs.in-project true
    poetry install
```
This will create a .venv/ folder directly in your project.

#### 5. 🔥 Lauch the application
```bash
cd App
cd Streamlit
streamlit run app.py
```






