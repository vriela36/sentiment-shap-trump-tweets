# Sentiment Analysis of Trump Tweets with SHAP

Projekt dotyczy analizy sentymentu tweetów z wykorzystaniem metod NLP oraz interpretacji działania modelu za pomocą metody SHAP.

Celem projektu jest przygotowanie danych tekstowych, zbudowanie modelu klasyfikującego sentyment tweetów oraz przeprowadzenie analizy wyjaśnialności modelu. Analiza SHAP pozwala sprawdzić, które słowa lub frazy miały największy wpływ na decyzje klasyfikatora.

## Dataset

W projekcie wykorzystano zbiór danych **Trump Tweets** dostępny w serwisie Kaggle:

https://www.kaggle.com/datasets/austinreese/trump-tweets

Użyty plik źródłowy:

```text
realdonaldtrump.csv
```

Zbiór zawiera tweety opublikowane przez konto `@realDonaldTrump`. W projekcie wykorzystano przede wszystkim kolumnę zawierającą treść tweeta, czyli `content`.

Ze względu na to, że dane źródłowe można pobrać bezpośrednio z Kaggle, w repozytorium nie musi znajdować się surowy plik `realdonaldtrump.csv`. W repozytorium umieszczony jest natomiast przetworzony plik danych po etapie preprocessingu:

```text
data/trump_tweets_preprocessed.csv
```

## Zakres projektu

Projekt obejmuje:

* przygotowanie i oczyszczenie danych tekstowych,
* automatyczne etykietowanie sentymentu metodą VADER,
* eksploracyjną analizę danych,
* budowę modelu klasyfikacji sentymentu,
* ocenę jakości modelu,
* analizę SHAP oraz interpretację wyników.

## Struktura projektu

```text
sentiment-shap-trump-tweets/
├── data/
│   ├── trump_tweets_preprocessed.csv
│   ├── sentiment_examples.csv
│   └── shap_top_features.csv
│
├── notebooks/
│   ├── 01_preprocessing_eda_sentiment_labels.ipynb
│   ├── 02_model_sentiment_classification.ipynb
│   └── 03_shap_analysis.ipynb
│
├── figures/
│   ├── sentiment_distribution.png
│   ├── tweet_length_distribution.png
│   ├── most_common_words.png
│   ├── wordcloud.png
│   ├── confusion_matrix.png
│   ├── shap_positive_bar.png
│   ├── shap_neutral_bar.png
│   └── shap_negative_bar.png
│
├── main.tex
├── main.pdf
├── requirements.txt
├── README.md
└── .gitignore
```

## Przygotowanie danych

W pierwszym etapie wczytano plik `realdonaldtrump.csv`, sprawdzono strukturę danych oraz wykonano preprocessing tekstu.

W ramach przygotowania danych:

* usunięto duplikaty w kolumnie `content`,
* przygotowano kolumnę `vader_text` do etykietowania metodą VADER,
* przygotowano kolumnę `clean_text` do dalszego modelowania,
* usunięto rekordy, dla których po czyszczeniu tekst był pusty,
* wyznaczono długość tekstu i liczbę słów,
* obliczono wynik sentymentu `vader_score`,
* przypisano etykietę `sentiment`.

Do etykietowania wykorzystano metodę VADER. Dla każdego tweeta obliczono wartość `compound`, a następnie przypisano jedną z trzech klas:

```text
compound >= 0.05  -> positive
compound <= -0.05 -> negative
pozostałe         -> neutral
```

Po preprocessingu uzyskano plik:

```text
data/trump_tweets_preprocessed.csv
```

Plik zawiera 42 998 rekordów oraz 14 kolumn.

## Kolumny w przygotowanym zbiorze

Plik `trump_tweets_preprocessed.csv` zawiera następujące kolumny:

* `id` — identyfikator tweeta,
* `date` — data publikacji tweeta,
* `content` — oryginalna treść tweeta,
* `vader_text` — lekko oczyszczony tekst użyty do etykietowania metodą VADER,
* `clean_text` — oczyszczony tekst przeznaczony do dalszego modelowania,
* `original_length` — długość oryginalnego tekstu,
* `clean_length` — długość tekstu po czyszczeniu,
* `word_count` — liczba słów po czyszczeniu,
* `retweets` — liczba retweetów,
* `favorites` — liczba polubień,
* `mentions` — wzmianki o użytkownikach,
* `hashtags` — hashtagi,
* `vader_score` — wynik `compound` z metody VADER,
* `sentiment` — przypisana etykieta sentymentu.

## Wyniki preprocessingu

Po wykonaniu preprocessingu uzyskano następujące liczby:

| Etap                                       | Liczba rekordów |
| ------------------------------------------ | --------------: |
| Dane początkowe                            |          43 352 |
| Po usunięciu duplikatów                    |          43 091 |
| Po usunięciu pustych wartości `clean_text` |          42 998 |

Rozkład klas sentymentu po etykietowaniu metodą VADER:

| Klasa    | Liczba tweetów | Udział |
| -------- | -------------: | -----: |
| positive |         25 038 | 58.23% |
| negative |         10 660 | 24.79% |
| neutral  |          7 300 | 16.98% |

## Wizualizacje

W katalogu `figures/` znajdują się wizualizacje przygotowane w ramach eksploracyjnej analizy danych:

* `sentiment_distribution.png` — rozkład klas sentymentu,
* `tweet_length_distribution.png` — rozkład długości tweetów po czyszczeniu,
* `most_common_words.png` — najczęściej występujące słowa,
* `wordcloud.png` — chmura słów,
* `confusion_matrix.png` — macierz pomyłek modelu,
* `shap_positive_bar.png` — najważniejsze cechy SHAP dla klasy pozytywnej,
* `shap_neutral_bar.png` — najważniejsze cechy SHAP dla klasy neutralnej,
* `shap_negative_bar.png` — najważniejsze cechy SHAP dla klasy negatywnej.

## Podział pracy

* Gabriela Froń — przygotowanie zbioru danych, oczyszczenie tweetów, etykiety sentymentu VADER oraz eksploracyjna analiza danych.
* Adam Balski — reprezentacja TF-IDF, model klasyfikacyjny oraz ocena jakości predykcji.
* Natalia Przychodzień — analiza SHAP, wizualizacje interpretacyjne, omówienie wyników oraz końcowe sprawozdanie.

## Ograniczenia etykietowania

Etykiety sentymentu zostały wygenerowane automatycznie metodą VADER, dlatego należy traktować je jako etykiety przybliżone. Metoda ta może popełniać błędy w przypadku nazw własnych, ironii, wieloznacznych słów oraz kontekstu politycznego lub kulturowego.

Przykładowo wyrażenia takie jak `Miss Universe` lub `Miss USA` mogą zostać błędnie potraktowane jako negatywne, ponieważ słowo `miss` w słowniku VADER ma ujemne nacechowanie. W analizowanych danych słowo to występuje jednak jako część nazwy własnej.

## Uruchomienie projektu

### 1. Klonowanie repozytorium

```bash
git clone <link-do-repozytorium>
cd sentiment-shap-trump-tweets
```

### 2. Instalacja zależności

```bash
pip install -r requirements.txt
```

### 3. Pobranie danych źródłowych

Należy pobrać zbiór danych z Kaggle:

https://www.kaggle.com/datasets/austinreese/trump-tweets

Następnie należy wykorzystać plik:

```text
realdonaldtrump.csv
```

### 4. Przygotowanie danych

Kod odpowiedzialny za przygotowanie danych znajduje się w notebooku:

```text
notebooks/01_preprocessing_eda_sentiment_labels.ipynb
```

Po uruchomieniu notebooka tworzony jest plik:

```text
data/trump_tweets_preprocessed.csv
```

### 5. Budowa modelu

Kod odpowiedzialny za budowę modelu klasyfikacyjnego znajduje się w notebooku:

```text
notebooks/02_model_sentiment_classification.ipynb
```

### 6. Analiza SHAP

Kod odpowiedzialny za analizę SHAP znajduje się w notebooku:

```text
notebooks/03_shap_analysis.ipynb
```

## Wykorzystane biblioteki

W projekcie wykorzystano między innymi:

* `pandas`,
* `numpy`,
* `matplotlib`,
* `vaderSentiment`,
* `wordcloud`,
* `scikit-learn`,
* `shap`.

## Status projektu

Projekt zawiera preprocessing, model klasyfikacyjny, ocenę jakości, analizę SHAP w notebookach Jupyter.
