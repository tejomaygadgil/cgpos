Part-of-Speech Tagging for Classical Greek 🏺
==============================
This project implements a [part-of-speech tagger](https://en.wikipedia.org/wiki/Part-of-speech_tagging) for Ancient Classical Greek, a [morphologically rich](https://arxiv.org/pdf/2005.01330.pdf) and [low-resource](https://arxiv.org/pdf/2006.07264.pdf) language, using texts from [The Ancient Greek and Latin Dependency Treebank](https://perseusdl.github.io/treebank_data/) and ideas inspired by [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/).

A syllable-based [Naive Bayes model](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) with [Laplace/Lidstone smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) is implemented to predict fine-grained part-of-speech using likelihood and prior estimates from the training data. A variant called `StupidBayes` (due to its relation to the [Stupid Backoff](https://aclanthology.org/D07-1090.pdf) smoothing method) is introduced that offers similar accuracy with faster performance.

This morphological approach addresses a major difficulty of part-of-speeching tagging Ancient Greek using [classical methods](https://en.wikipedia.org/wiki/Hidden_Markov_model): namely, the [complex system of word endings](https://en.wiktionary.org/wiki/Appendix:Ancient_Greek_grammar_tables) that results in many [singularly occurring words](https://en.wikipedia.org/wiki/Hapax_legomenon#Ancient_Greek_examples) and a highly flexible word order within sentences[^1].

# Implementation
## Multinomial Naive Bayes
[`MultinomialNaiveBayes`](https://github.com/tejomaygadgil/cgpos/blob/9e49c0872ff4146b824521cf7c506ec3465e9ea5/src/cgpos/model/multinomial_naive_bayes.py#L14C11-L14C11)[^2] is trained on n-grams of word syllables generated from [The Ancient Greek and Latin Dependency Treebank](https://perseusdl.github.io/treebank_data/). N-gram depth is controllable via the `ngram_range` parameter, with `ngram_range=(1, 5)` providing the best performance on the development set (see below). The first training pass counts the occurrence of syllables per category, as well as class occurrences. [Greek diacritics](https://en.wikipedia.org/wiki/Greek_diacritics), which are usually stripped, are preserved to give more information to the model.

The second pass normalizes the raw counts to produce probabilities (represented via [log probabilities](https://en.wikipedia.org/wiki/Log_probability) for numerical stability) for syllable likelihoods and class priors. [Laplace/Lidstone smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) is implemented using an `alpha` parameter that can be adjusted at runtime, with `alpha=0.2` giving best results on the development set (see below). As likelihood probability distributions are sparse, dictionaries and default values are used to speed up computation.

At prediction time the model uses [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) (assuming [conditional independence](https://en.wikipedia.org/wiki/Conditional_independence#Uses_in_Bayesian_inference)) to estimate the most likely class using the following relationship:

$$\begin{align*} 
\text{argmax}_c \log P(\text{class}_c|\text{syllables}) &= \text{argmax}_c \sum_i  \log P(\text{syllable}_i|\text{class}_c)  + \log P(\text{class}_c)
\end{align*} $$

## Stupid Bayes
[`StupidBayes`](https://github.com/tejomaygadgil/cgpos/blob/9e49c0872ff4146b824521cf7c506ec3465e9ea5/src/cgpos/model/multinomial_naive_bayes.py#L102)[^3], on the other hand, is a variant of `MultinomialNaiveBayes` that skips probabilities altogether. The training pass only stores occurrence of syllables per category. A simplified version of [n-grams backoff](https://en.wikipedia.org/wiki/Katz%27s_back-off_model) is implemented to only generate shorter n-grams in order to fill in lookup gaps. 

Predictions are generated during test time by simply returns the class with the highest count amongst all the input n-grams. Formally, this is given by: 

$$\begin{align*} 
\text{argmax}_c (\text{class}_c|\text{syllables}) &= \sum_i  C(\text{class}_c|\text{syllable}_i)
\end{align*} $$

# Results
## Training
Module training is carried out by [`cgpos.eval.train`](https://github.com/tejomaygadgil/cgpos/blob/main/src/cgpos/eval/train.py). Training and test sets are produced using a [shuffled split](https://scikit-learn.org/stable/modules/cross_validation.html#shufflesplit) strategy. Training is parallelized via [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures) to speed up training time. Results are stored in [`runs`]() with the format `YYYY-mm-dd_HH-MM-SS`.

[`cgpos.eval.eval`](https://github.com/tejomaygadgil/cgpos/blob/main/src/cgpos/eval/eval.py) carries out hyperparameter tuning and model selection according to [`config.yaml`](https://github.com/tejomaygadgil/cgpos/blob/f78e7ce3f9674ed7c9c44d666ac7cbef61a2f4fc/config/config.yaml#L22). [Stratified k-fold shuffle](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) strategy (`k=5`) and [F1 score evaluation metric](https://en.wikipedia.org/wiki/F-score) are employed to handle class imbalance issues.  Multioutput predictions are achieved by following the [simple strategy](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) of fitting one `StupidBayes` per target, and a [`PartOfSpeechTagger`](https://github.com/tejomaygadgil/cgpos/blob/main/src/cgpos/model/pos_tagger.py) is created based on the best model configuration.

The best model is stored in [`models/pos_tagger.pkl`](https://github.com/tejomaygadgil/cgpos/blob/main/models/pos_tagger.pkl) and a text report showing the best model configuration, classification reports and confusion matrices is saved to [`reports/report.txt`](https://github.com/tejomaygadgil/cgpos/blob/main/reports/report.txt).

## Performance
### Best model
**The overall fine-grained test accuracy for the best model after hyperparameter tuning is 80.19%.**

This is the best model architecture after hyperparameter tuning :
```
     pos: ('StupidBayes', {'ngram_depth': 9}),
  person: ('MultinomialNaiveBayes', {'alpha': 0.5, 'ngram_range': (1, 2)}),
  number: ('StupidBayes', {'ngram_depth': 6}),
  gender: ('MultinomialNaiveBayes', {'alpha': 0.1, 'ngram_range': (1, 3)}),
    case: ('StupidBayes', {'ngram_depth': 7}),
   tense: ('StupidBayes', {'ngram_depth': 7}),
    mood: ('MultinomialNaiveBayes', {'alpha': 0.3, 'ngram_range': (1, 2)}),
   voice: ('MultinomialNaiveBayes', {'alpha': 0.8, 'ngram_range': (1, 5)})}
  degree: ('StupidBayes', {'ngram_depth': 9}),
```

### Classification reports
#### Part of speech
```
              precision    recall  f1-score   support

         N/A       0.01      0.33      0.02         3
     article       0.89      0.86      0.87      2983
        noun       0.94      0.89      0.91     11732
   adjective       0.81      0.90      0.85      5316
     pronoun       0.72      0.84      0.78      2921
        verb       0.94      0.89      0.91     10179
      adverb       0.57      0.89      0.70      3089
  adposition       0.96      0.91      0.93      2885
 conjunction       0.81      0.72      0.76      3133
     numeral       0.83      0.63      0.72        71
interjection       1.00      1.00      1.00        19
    particle       0.89      0.72      0.80      4810
 punctuation       1.00      0.99      1.00      5574
    accuracy                           0.87     52715
   macro avg       0.80      0.81      0.79     52715
weighted avg       0.89      0.87      0.87     52715
```

#### Person
```
               precision    recall  f1-score   support

          N/A       0.99      0.99      0.99     46656
 first person       0.76      0.77      0.77       839
second person       0.65      0.64      0.64       732
 third person       0.93      0.87      0.90      4488

     accuracy                           0.97     52715
    macro avg       0.83      0.82      0.83     52715
 weighted avg       0.97      0.97      0.97     52715
```

#### Number
```
              precision    recall  f1-score   support

         N/A       0.96      0.98      0.97     20844
    singular       0.96      0.92      0.94     22222
      plural       0.87      0.92      0.90      9516
        dual       0.64      0.92      0.76       133
    
    accuracy                           0.94     52715
   macro avg       0.86      0.94      0.89     52715
weighted avg       0.94      0.94      0.94     52715
```

#### Gender
```
              precision    recall  f1-score   support

         N/A       0.98      0.97      0.97     27309
   masculine       0.90      0.91      0.90     13571
    feminine       0.88      0.90      0.89      6704
      neuter       0.79      0.79      0.79      5131

    accuracy                           0.93     52715
   macro avg       0.89      0.89      0.89     52715
weighted avg       0.93      0.93      0.93     52715
```

#### Case
```
              precision    recall  f1-score   support

         N/A       0.98      0.95      0.96     27589
  nominative       0.80      0.87      0.83      7039
    genitive       0.92      0.92      0.92      4870
      dative       0.88      0.93      0.91      3628
  accusative       0.88      0.85      0.86      9309
    vocative       0.55      0.87      0.68       280

    accuracy                           0.92     52715
   macro avg       0.83      0.90      0.86     52715
weighted avg       0.92      0.92      0.92     52715
```

#### Tense-aspect
```
                 precision    recall  f1-score   support

            N/A       0.99      0.97      0.98     44010
        present       0.84      0.93      0.88      3471
      imperfect       0.78      0.92      0.84      1083
        perfect       0.81      0.90      0.85       509
plusquamperfect       0.67      0.59      0.63       105
 future perfect       0.00      0.00      0.00         1
         future       0.62      0.84      0.71       299
         aorist       0.85      0.91      0.88      3237

       accuracy                           0.96     52715
      macro avg       0.70      0.76      0.72     52715
   weighted avg       0.97      0.96      0.96     52715
```

#### Mood
```
              precision    recall  f1-score   support

         N/A       0.98      0.99      0.99     42640
  indicative       0.92      0.92      0.92      4707
 subjunctive       0.78      0.66      0.71       479
  infinitive       0.99      0.87      0.92      1372
  imperative       0.54      0.61      0.57       276
  participle       0.95      0.89      0.92      2922
    optative       0.89      0.72      0.80       319

    accuracy                           0.97     52715
   macro avg       0.87      0.81      0.83     52715
weighted avg       0.97      0.97      0.97     52715
```

#### Voice
```
               precision    recall  f1-score   support

          N/A       0.99      0.99      0.99     42987
       active       0.94      0.94      0.94      6765
      passive       0.77      0.91      0.84       261
       middle       0.84      0.81      0.83       902
medio-passive       0.93      0.84      0.89      1800

     accuracy                           0.98     52715
    macro avg       0.89      0.90      0.90     52715
 weighted avg       0.98      0.98      0.98     52715
```

#### Degree
```
              precision    recall  f1-score   support

         N/A       1.00      1.00      1.00     52532
    positive       0.00      0.00      0.00         1
 comparative       0.76      0.64      0.69       137
 superlative       0.37      0.67      0.48        45

    accuracy                           1.00     52715
   macro avg       0.53      0.58      0.54     52715
weighted avg       1.00      1.00      1.00     52715
```

### Confusion matrix
| true / pred  |   article |   noun |   adjective |   pronoun |   verb |   adverb |   adposition |   conjunction |   numeral |   interjection |   particle |   punctuation |
|:-------------|----------:|-------:|------------:|----------:|-------:|---------:|-------------:|--------------:|----------:|---------------:|-----------:|--------------:|
| article      |      2559 |      3 |           4 |       405 |      5 |        7 |            0 |             0 |         0 |              0 |          0 |             0 |
| noun         |        63 |  10393 |         491 |       179 |    404 |       73 |            9 |            33 |         1 |              0 |         63 |             0 |
| adjective    |         4 |    253 |        4763 |       160 |     74 |       21 |            9 |            15 |         7 |              0 |          2 |             0 |
| pronoun      |        69 |     20 |         302 |      2456 |     15 |       23 |            0 |             1 |         0 |              0 |         34 |             0 |
| verb         |         1 |    326 |         242 |       100 |   9031 |       64 |           39 |           151 |         1 |              0 |        211 |             0 |
| adverb       |         4 |     38 |          36 |        24 |     45 |     2744 |           54 |            65 |         0 |              0 |         74 |             0 |
| adposition   |       160 |      3 |           5 |        17 |     12 |       50 |         2635 |             2 |         0 |              0 |          0 |             0 |
| conjunction  |         4 |      2 |           4 |        57 |      3 |      763 |            5 |          2252 |         0 |              0 |         42 |             0 |
| numeral      |         0 |      1 |          25 |         0 |      0 |        0 |            0 |             0 |        45 |              0 |          0 |             0 |
| interjection |         0 |      0 |           0 |         0 |      0 |        0 |            0 |             0 |         0 |             19 |          0 |             0 |
| particle     |         7 |      7 |          11 |         4 |     24 |     1033 |            6 |           252 |         0 |              0 |       3464 |             0 |
| punctuation  |         0 |      0 |           0 |         0 |      0 |        0 |            0 |             0 |         0 |              0 |          0 |          5545 |

As expected, the part-of-speech tagger does well on classifying highly morphologically-driven elements such as nouns and verbs, but struggles to addressing positionally-driven elements such as conjugations or particles. Some kind of positional augmentation (provided via models such as [Conditional Random Fields](https://en.wikipedia.org/wiki/Conditional_random_field)) would be necessary to address this.  

# Build instructions
## 1. Set environment
Get project requirements by setting up a `poetry` environment:
1. Install [poetry](https://python-poetry.org/docs/#installation)
2. In the terminal, run:
```
$ git clone https://github.com/tejomaygadgil/cgpos.git
$ cd cgpos/
$ make activate_poetry
poetry install
Installing dependencies from lock file

No dependencies to install or update

Installing the current project: cgpos (1.0.0)
...
```

## 2. Get data
Grab project data:
```
$ cd /dir/to/repository
$ make make_dataset
Initializing data directory
mkdir data/raw/zip
Grabbing Perseus data
...
```

## 3. Build features
Ready data for training by tokenizing word syllables: 
```
$ cd /dir/to/repository
$ make build_features
Building features
python src/cgpos/data/features.py
...
```

## 4. Train model
Train the part-of-speech tagger and evaluate performance using: 
```
$ cd /dir/to/repository
$ make train_model
Training model
python src/cgpos/eval/train.py
...
```

Hyperparameter range and model selection are adjustable via [config file](https://github.com/tejomaygadgil/cgpos/blob/main/conf/main.yaml).

# References
## Help
You can get a helpfile of all available `make` options by running:
```
$ make
Available rules:

activate_poetry     Activate poetry environment 
build_features      Build features 
get_data            Get raw data 
init_data_dir       Initialize data directory 
install_poetry      Install poetry environment 
make_dataset        Make Perseus dataset 
remove_all_data     Remove all data 
remove_data         Remove processed data 
tests               Run tests
train_model         Train model and evaluate performance

```

## Tools
This repository uses the following tools:
* [`make`](https://www.gnu.org/software/make/) for running code
* [`poetry`](https://python-poetry.org) for package management 
* [`hydra`](https://hydra.cc/) for code reproducibility
* [`black`](https://github.com/psf/black) and [`ruff`](https://github.com/charliermarsh/ruff-pre-commit) for code review 

[^1]: This is in contrast to [analytic languages](https://en.wikipedia.org/wiki/Analytic_language) like English that use word order and special words like "had" and "will" to express part-of-speech.

[^2]: `from cgpos.models.multinomial_naive_bayes import MultinomialNaiveBayes`

[^3]: `from cgpos.models.multinomial_naive_bayes import StupidBayes`
