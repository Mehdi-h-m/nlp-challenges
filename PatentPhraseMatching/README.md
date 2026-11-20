# U.S. Patent Phrase to Phrase Matching

Predict a similarity score between 0 and 1 for two phrases in a Cooperative Patent Classification category. DistilBERT takes the anchor and CPC context alongside the target phrase and predicts the similarity score. The [Kaggle competition](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching) evaluates predictions with Pearson correlation.

The validation split groups rows by normalized anchor phrase so the same anchor stays on one side of the split. The script chooses the epoch with the best validation Pearson correlation and predicts a score for each test ID.

## Run locally

Install PyTorch 2.6 or later for your hardware and install the remaining dependencies with `python -m pip install pandas numpy scikit-learn transformers`. Download and unzip the [competition data](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/data) into this folder's `data/` directory. Run this from the workspace root.

~~~bash
python PatentPhraseMatching/train.py
~~~

The default output is `PatentPhraseMatching/submissions/submission.csv`. The local test file is a small example. Kaggle supplies the evaluation test set when it scores the submitted notebook.

## Run on Kaggle

From the workspace root, replace `YOUR_KAGGLE_USERNAME` with your Kaggle username and run the commands below. The notebook attaches the competition data and an offline DistilBERT checkpoint.

~~~bash
python package_kaggle.py YOUR_KAGGLE_USERNAME
kaggle kernels push -p runs/kaggle/patent-phrase-matching-distilbert --accelerator NvidiaTeslaT4
~~~

Once the notebook finishes, replace `VERSION` with the notebook version printed by the push command.

~~~bash
kaggle competitions submit us-patent-phrase-to-phrase-matching -k YOUR_KAGGLE_USERNAME/patent-phrase-matching-distilbert -v VERSION -f submission.csv -m "DistilBERT regression baseline"
~~~
