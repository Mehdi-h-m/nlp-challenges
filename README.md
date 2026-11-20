# NLP Challenges

My workspace for solving natural language processing challenges on Kaggle. Each challenge has its own folder for exploratory analysis, modeling notebooks, and evaluation figures.

## Challenges

| Challenge | Task | Progress |
| --- | --- | --- |
| [Jigsaw Toxic Comment Classification](Jiggsaw/) | Predict six overlapping toxicity labels for Wikipedia comments | EDA and TF-IDF baseline complete. DistilBERT training is in progress. |

## Jigsaw Toxic Comment Classification

The target labels are `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. A comment can have multiple labels. The notebooks compare models using mean per-label ROC AUC.

| Notebook | What it covers |
| --- | --- |
| [01 · Exploratory data analysis](Jiggsaw/notebooks/01_eda_visualize.ipynb) | Class imbalance, label co-occurrence, comment lengths, and common terms. |
| [02 · TF-IDF + logistic regression](Jiggsaw/notebooks/02_baseline_tfidf_logreg.ipynb) | Word unigrams and bigrams, one classifier per label, held-out validation, and Kaggle submission. |
| [04 · DistilBERT fine-tuning](Jiggsaw/notebooks/04_distilbert_finetune.ipynb) | Fine-tunes `distilbert-base-uncased` for multi-label classification and compares each label with the baseline. |

The TF-IDF baseline recorded **0.9812 mean ROC AUC** on its held-out validation split. Duplicate or near-duplicate comments may make that estimate optimistic. The [figures folder](Jiggsaw/figures/) includes the per-label baseline results and plots from the exploratory analysis.

See the [Kaggle competition](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) for the task and data. Rob Mulla’s [Kaggle EDA notebook](https://www.kaggle.com/code/robikscube/jiggsaw-toxic-comments-eda-twitch-stream/notebook) is a reference for the challenge.

## Run the notebooks

1. Download the competition data from the [Kaggle data page](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data). Extract `train.csv` and `test.csv` into `Jiggsaw/data/`.
2. In a Python environment, install the notebook and baseline dependencies.

   ```bash
   python -m pip install jupyter pandas numpy matplotlib seaborn scikit-learn wordcloud
   ```

3. For DistilBERT, also install PyTorch and Transformers in that environment. Use a CUDA-enabled PyTorch build if training on an NVIDIA GPU.
4. From `Jiggsaw/`, start Jupyter and open the notebooks under `notebooks/`.

   ```bash
   python -m jupyter notebook
   ```

5. The modeling notebooks write prediction CSVs to `Jiggsaw/submissions/`. Run the optional upload cells only if you want to submit to Kaggle and have configured the Kaggle CLI.

Some notebook outputs contain offensive language from the dataset.
