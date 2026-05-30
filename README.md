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

Ze względu na to, że dane źródłowe można pobrać bezpośrednio z Kaggle, w repozytorium nie musi znajdować się surowy plik `realdonaldtrump.csv`. W repozytorium umieszczony zostanie natomiast przetworzony plik danych po etapie preprocessingu:

```text
data/trump_tweets_preprocessed.csv
```

## Struktura projektu

```text
sentiment-shap-trump-tweets/
├── data/
│   ├── README.md
│   └── trump_tweets_preprocessed.csv
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
│   ├── confusion_matrix.png
│   ├── shap_positive_bar.png
│   └── shap_negative_bar.png
│
├── report/
│   └── report.pdf
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Etap 1: przygotowanie danych

W pierwszym etapie wykonano przygotowanie danych tekstowych do dalszej analizy. Obejmowało ono:

* wczytanie danych z pliku `realdonaldtrump.csv`,
* wybór kolumny zawierającej treść tweetów,
* usunięcie braków danych,
* usunięcie duplikatów,
* oczyszczenie tekstu tweetów,
* obliczenie długości tweetów,
* automatyczne przypisanie etykiet sentymentu,
* wykonanie eksploracyjnej analizy danych.

Czyszczenie tekstu obejmowało między innymi:

* zamianę tekstu na małe litery,
* usunięcie linków,
* usunięcie oznaczeń użytkowników,
* usunięcie symboli hashtagów,
* usunięcie znaków specjalnych,
* usunięcie nadmiarowych spacji.

Ponieważ zbiór danych nie zawierał ręcznie przygotowanych etykiet sentymentu, zastosowano metodę VADER. Dla każdego tweeta obliczono wartość `compound score`, a następnie przypisano jedną z trzech klas:

```text
compound >= 0.05  -> positive
compound <= -0.05 -> negative
pozostałe         -> neutral
```

Wynikiem tego etapu jest plik:

```text
data/trump_tweets_preprocessed.csv
```

Plik zawiera między innymi kolumny:

* `content` — oryginalna treść tweeta,
* `clean_text` — oczyszczona treść tweeta,
* `original_length` — długość oryginalnego tekstu,
* `clean_length` — długość tekstu po czyszczeniu,
* `word_count` — liczba słów po czyszczeniu,
* `vader_score` — wynik sentymentu VADER,
* `sentiment` — etykieta sentymentu: `positive`, `neutral` albo `negative`.

## Etap 2: model klasyfikacji sentymentu

W drugim etapie przygotowany tekst zostanie przekształcony do postaci numerycznej z wykorzystaniem metody TF-IDF. Następnie zostanie zbudowany model klasyfikacji sentymentu.

Planowane podejście:

* podział danych na zbiór treningowy i testowy,
* reprezentacja tekstu metodą TF-IDF,
* trenowanie modelu klasyfikacyjnego,
* ocena jakości modelu z wykorzystaniem metryk klasyfikacyjnych.

Do oceny modelu zostaną wykorzystane między innymi:

* accuracy,
* precision,
* recall,
* F1-score,
* macierz pomyłek.

Jako główny model planowane jest użycie `Logistic Regression`, ponieważ jest to model dobrze współpracujący z reprezentacją TF-IDF oraz łatwy do interpretacji przy pomocy SHAP.

## Etap 3: analiza SHAP

W trzecim etapie zostanie przeprowadzona analiza SHAP dla wytrenowanego modelu. Celem tej analizy będzie sprawdzenie, które słowa i frazy miały największy wpływ na klasyfikację sentymentu.

Analiza obejmie:

* globalną interpretację modelu,
* wskazanie najważniejszych cech dla klasy pozytywnej,
* wskazanie najważniejszych cech dla klasy negatywnej,
* lokalną interpretację wybranych tweetów,
* analizę przykładów poprawnie i błędnie sklasyfikowanych.

Wizualizacje SHAP pozwolą pokazać, czy model podejmuje decyzje na podstawie słów rzeczywiście związanych z emocjonalnym nacechowaniem tekstu, czy raczej opiera się na często występujących słowach specyficznych dla badanego zbioru danych.

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

### 3. Przygotowanie danych

Należy pobrać zbiór danych z Kaggle:

https://www.kaggle.com/datasets/austinreese/trump-tweets

Następnie należy pobrać i wykorzystać plik:

```text
realdonaldtrump.csv
```

Plik można wczytać w notebooku:

```text
notebooks/01_preprocessing_eda_sentiment_labels.ipynb
```

Po uruchomieniu notebooka zostanie wygenerowany plik:

```text
data/trump_tweets_preprocessed.csv
```

### 4. Trenowanie modelu

Model klasyfikacyjny znajduje się w notebooku:

```text
notebooks/02_model_sentiment_classification.ipynb
```

### 5. Analiza SHAP

Analiza wyjaśnialności modelu znajduje się w notebooku:

```text
notebooks/03_shap_analysis.ipynb
```

## Wykorzystane biblioteki

W projekcie wykorzystano między innymi:

* `pandas`,
* `numpy`,
* `matplotlib`,
* `scikit-learn`,
* `vaderSentiment`,
* `wordcloud`,
* `shap`.

## Autorzy i podział pracy

Projekt został wykonany przez zespół trzyosobowy.

| Osoba   | Zakres prac                                                                                              |
| ------- | -------------------------------------------------------------------------------------------------------- |
| Osoba 1 | Przygotowanie danych, preprocessing, etykietowanie sentymentu metodą VADER, eksploracyjna analiza danych |
| Osoba 2 | Budowa modelu klasyfikacji sentymentu, reprezentacja tekstu metodą TF-IDF, ocena jakości modelu          |
| Osoba 3 | Analiza SHAP, interpretacja wyników, przygotowanie wizualizacji i końcowych wniosków                     |

## Status projektu

Projekt w trakcie realizacji.

Aktualnie realizowany etap:

```text
Etap 1: przygotowanie danych, preprocessing i analiza wstępna
```
