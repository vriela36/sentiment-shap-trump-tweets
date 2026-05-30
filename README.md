# Sentiment Analysis of Trump Tweets with SHAP

Projekt dotyczy analizy sentymentu tweetów z wykorzystaniem metod NLP oraz interpretacji modelu za pomocą metody SHAP.

## Dataset

W projekcie wykorzystano zbiór danych Trump Tweets dostępny w serwisie Kaggle:

https://www.kaggle.com/datasets/austinreese/trump-tweets

Użyty plik źródłowy: `realdonaldtrump.csv`.

Ze względu na to, że dane źródłowe można pobrać bezpośrednio z Kaggle, w repozytorium przechowywany jest przede wszystkim przetworzony plik danych po etapie preprocessingu.

## Struktura projektu

```text
data/
  README.md
  trump_tweets_preprocessed.csv

notebooks/
  01_preprocessing_eda_sentiment_labels.ipynb

figures/
  sentiment_distribution.png
  tweet_length_distribution.png
  most_common_words.png

requirements.txt
