# NLP Challenges

My workspace for Kaggle NLP challenges. It covers multi-label classification, phrase similarity, and text span extraction.

## Results

| Challenge | Model | Metric | Validation | Kaggle public | Kaggle private | Submission ID |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| [Jigsaw Toxic Comment Classification](#jigsaw-toxic-comment-classification) | TF-IDF + logistic regression | Mean ROC AUC | 0.9812 | 0.97550 | 0.97577 | `58622163` |
| [Jigsaw Toxic Comment Classification](#jigsaw-toxic-comment-classification) | DistilBERT classification | Mean ROC AUC | 0.9906 | 0.98404 | 0.98449 | `58634677` |
| [U.S. Patent Phrase to Phrase Matching](PatentPhraseMatching/) | DistilBERT regression | Pearson correlation | 0.76278 | 0.7339 | 0.7507 | `58635351` |
| [Tweet Sentiment Extraction](TweetSentimentExtraction/) | DistilBERT span extraction | Mean word-level Jaccard | 0.69082 | 0.69202 | 0.69449 | `58635350` |

Patent Matching and Tweet Sentiment have local training and submission instructions in their project READMEs. The packaging script builds their Kaggle notebooks and attaches the model weights as an offline input. [Patent Matching](PatentPhraseMatching/) · [Tweet Sentiment](TweetSentimentExtraction/) · [Kaggle packaging script](package_kaggle.py)

## Jigsaw Toxic Comment Classification

The target labels are `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. A comment can have multiple labels. The notebooks compare models using mean per-label ROC AUC.

| Notebook | What it covers |
| --- | --- |
| [01 · Exploratory data analysis](Jiggsaw/notebooks/01_eda_visualize.ipynb) | Class imbalance, label co-occurrence, comment lengths, and common terms. |
| [02 · TF-IDF + logistic regression](Jiggsaw/notebooks/02_baseline_tfidf_logreg.ipynb) | Word unigrams and bigrams, one classifier per label, held-out validation, and Kaggle submission. |
| [04 · DistilBERT fine-tuning](Jiggsaw/notebooks/04_distilbert_finetune.ipynb) | Fine-tunes `distilbert-base-uncased` for multi-label classification and compares each label with the baseline. |

Duplicate or near-duplicate comments across the training and validation split may make the validation scores optimistic. The figures include per-label results for the TF-IDF baseline and plots from exploratory analysis. [Jigsaw figures](Jiggsaw/figures/)

See the [Kaggle competition](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) for the task and data. Rob Mulla’s [Kaggle EDA notebook](https://www.kaggle.com/code/robikscube/jiggsaw-toxic-comments-eda-twitch-stream/notebook) is a reference for the challenge.

### Run the Jigsaw notebooks

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
