Part-of-Speech Tagging for Classical Greek 🏺
==============================
This project implements a [part-of-speech tagger](https://en.wikipedia.org/wiki/Part-of-speech_tagging) for Ancient Classical Greek, a [morphologically rich](https://arxiv.org/pdf/2005.01330.pdf) and [low-resource](https://arxiv.org/pdf/2006.07264.pdf) language, using texts from [The Ancient Greek and Latin Dependency Treebank](https://perseusdl.github.io/treebank_data/) and ideas inspired by [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/).

A syllable-based [Naive Bayes model](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) with [Laplace/Lidstone smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) is implemented to predict fine-grained part-of-speech using likelihood and prior estimates from the training data. A variant called `StupidBayes` (due to its relation to the [Stupid Backoff](https://aclanthology.org/D07-1090.pdf) smoothing method) is introduced that offers higher accuracy and 5-10x faster performance.

This morphological approach addresses a major difficulty of part-of-speeching tagging Ancient Greek using [classical methods](https://en.wikipedia.org/wiki/Hidden_Markov_model): namely, the [complex system of word endings](https://en.wiktionary.org/wiki/Appendix:Ancient_Greek_grammar_tables)[^1] that results in many [singularly occurring](https://en.wikipedia.org/wiki/Hapax_legomenon#Ancient_Greek_examples) words and highly flexible word order[^2].

# Implementation
## Multinomial Naive Bayes
`MultinomialNaiveBayes`[^3] is trained on n-grams of word syllables generated from [The Ancient Greek and Latin Dependency Treebank](https://perseusdl.github.io/treebank_data/). N-gram depth is controllable via the `ngram_range` parameter, with `ngram_range=(1, 5)` providing the best performance on the development set (see below). The first training pass counts the occurrence of syllables per category, as well as class occurrences. [Greek diacritics](https://en.wikipedia.org/wiki/Greek_diacritics), which are usually stripped, are preserved to give more information to the model.

The second pass normalizes the raw counts to produce probabilities (represented via [log probabilities](https://en.wikipedia.org/wiki/Log_probability) for numerical stability) for syllable likelihoods and class priors. [Laplace/Lidstone smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) is implemented using an `alpha` parameter that can be adjusted at runtime, with `alpha=0.2` giving best results on the development set (see below). As likelihood probability distributions are sparse, dictionaries and default values are used to speed up computation.

At prediction time the model uses [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) (and an assumption of [conditional independence](https://en.wikipedia.org/wiki/Conditional_independence#Uses_in_Bayesian_inference)) to estimate the most likely class using the following relationship:

$$\begin{align*} 
\text{argmax}_c \log P(\text{class}_c|\text{syllables}) &= \text{argmax}_c \sum_i  \log P(\text{syllable}_i|\text{class}_c)  + \log P(\text{class}_c)
\end{align*} $$

Multioutput predictions are achieved by following the [simple strategy](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) of fitting one `MultinomialNaiveBayes` per target.

## Stupid Bayes
`StupidBayes`[^4], on the other hand, is a variant of `MultinomialNaiveBayes` that skips probabilities altogether. The training pass only stores occurrence of syllables per category. A simplified version of [n-grams backoff](https://en.wikipedia.org/wiki/Katz%27s_back-off_model) is implemented to only generate shorter n-grams in order to fill in lookup gaps. 

Predictions are generated during test time by simply returns the class with the highest count amongst all the input n-grams. Formally, this is given by: 

$$\begin{align*} 
\text{argmax}_c (\text{class}_c|\text{syllables}) &= \sum_i  C(\text{class}_c|\text{syllable}_i)
\end{align*} $$

Multioutput predictions are achieved by following the [simple strategy](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) of fitting one `StupidBayes` per target.

# Results




# Running the code
## Instructions
### 1. Set environment
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

Installing the current project: cgpos (0.1.0)
...
```

### 2. Get data
Grab project data:
```
$ cd /dir/to/repository
$ make make_dataset
Initializing data directory
mkdir data/raw/zip
Grabbing Perseus data
...
```

### 3. Run model
Train the part-of-speech tagger using: 
```
$ cd /dir/to/repository
$ make run_model # TODO
Building features
python src/cgpos/features/build_features.py
...
```


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

```

## Tools
This repository uses the following tools:
* [`make`](https://www.gnu.org/software/make/) for running code
* [`poetry`](https://python-poetry.org) for package management 
* [`hydra`](https://hydra.cc/) for code reproducibility
* [`black`](https://github.com/psf/black) and [`ruff`](https://github.com/charliermarsh/ruff-pre-commit) for code review 

[^1]: This is in contrast to [analytic languages](https://en.wikipedia.org/wiki/Analytic_language) like English that use word order and special words like "had" and "will" to express part-of-speech.

[^2]: This feature in particular poses a challenge to [classical methods that heavily rely on word order](https://en.wikipedia.org/wiki/Hidden_Markov_model) to make predictions.

[^3]: `from cgpos.models.multinomial_naive_bayes import MultinomialNaiveBayes`

[^4]: `from cgpos.models.multinomial_naive_bayes import StupidBayes`
