# Tweet Sentiment Extraction

Extract the substring of a tweet that best supports its given sentiment. The [Kaggle competition](https://www.kaggle.com/competitions/tweet-sentiment-extraction) scores predictions using mean word-level Jaccard similarity.

DistilBERT predicts start and end tokens for positive and negative tweets. The model uses tokenizer offsets to slice the original text, preserving punctuation and casing. For neutral tweets, it predicts the full tweet. Validation uses a sentiment-stratified split and includes all three sentiments when scoring.

## Run locally

Install PyTorch 2.6 or later for your hardware and install the remaining dependencies with `python -m pip install pandas numpy scikit-learn transformers`. Download and unzip the [competition data](https://www.kaggle.com/competitions/tweet-sentiment-extraction/data) into this folder's `data/` directory. Run this from the workspace root.

~~~bash
python TweetSentimentExtraction/train.py
~~~

The default output is `TweetSentimentExtraction/submissions/submission.csv`.

## Run on Kaggle

From the workspace root, replace `YOUR_KAGGLE_USERNAME` with your Kaggle username and run the commands below. The notebook attaches the competition data and an offline DistilBERT checkpoint.

~~~bash
python package_kaggle.py YOUR_KAGGLE_USERNAME
kaggle kernels push -p runs/kaggle/tweet-sentiment-extraction-distilbert --accelerator NvidiaTeslaT4
~~~

Once the notebook finishes, replace `VERSION` with the notebook version printed by the push command.

~~~bash
kaggle competitions submit tweet-sentiment-extraction -k YOUR_KAGGLE_USERNAME/tweet-sentiment-extraction-distilbert -v VERSION -f submission.csv -m "DistilBERT sentiment span baseline"
~~~
