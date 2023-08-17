Part-of-Speech Tagging for Classical Greek 🏺
==============================
This project implements a [part-of-speech tagger](https://en.wikipedia.org/wiki/Part-of-speech_tagging) for Ancient Classical Greek, a [morphologically rich](https://arxiv.org/pdf/2005.01330.pdf) and [low-resource](https://arxiv.org/pdf/2006.07264.pdf) language, using texts from [The Ancient Greek and Latin Dependency Treebank](https://perseusdl.github.io/treebank_data/) and ideas inspired by [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/).

# Model
A Syllable-Based [Naive Bayes model](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) with [Laplace/Lidstone smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) is implemented to predict fine-grained part-of-speech using likelihood and prior estimates from the training data. A variant called `StupidBayes` (due to its relation to the [Stupid Backoff](https://aclanthology.org/D07-1090.pdf) smoothing method) is introduced that delivers a 5-10% improvement in accuracy.

This purely morphological approach overcomes the main difficulty of using [classical methods](https://en.wikipedia.org/wiki/Hidden_Markov_model) to parse Ancient Greek: a [complex system of word endings](https://en.wiktionary.org/wiki/Appendix:Ancient_Greek_grammar_tables) to indicate part-of-speech[^1]. 

## Implementation

`MultinomialNaiveBayes`[^2] is trained on n-grams of word syllables generated from [The Ancient Greek and Latin Dependency Treebank](https://perseusdl.github.io/treebank_data/). N-gram depth is controllable via the `ngram_range` parameter, with `ngram_range=(1, 5)` providing the best performance on the development set (see below). The first training pass counts the occurence of syllables per category (likelihoods), as well as class occurences (priors). [Greek diacritics](https://en.wikipedia.org/wiki/Greek_diacritics), which are usually stripped, are preserved to give more information to the model. 

The second pass normalizes the raw counts into probabilities (represented via [log probabilities](https://en.wikipedia.org/wiki/Log_probability) numerical stability). [Laplace/Lidstone smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) is implemented using an `alpha` parameter that can be adjusted at runtime, with `alpha=0.2` giving best results on the development split (see below).

At test time the model uses [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) with a [conditional independence](https://en.wikipedia.org/wiki/Conditional_independence#Uses_in_Bayesian_inference) assumption to estimate $\text{argmax}_c P(\text{class}_c|\text{syllables})$ from the following relationship:

$$\begin{align*} 
\text{argmax}_c \log P(\text{class}_c|\text{syllables}) &= \text{argmax}_c \sum_i  \log P(\text{syllable}_i|\text{class}_c)  + P(\text{class}_c)
\end{align*} $$

`StupidBayes`[^3]

## Results




[^1]: This is in contrast to [analytics languages](https://en.wikipedia.org/wiki/Analytic_language) like English that use word order and special words like "had" and "will" to express part-of-speech.

[^2]: `from cgpos.models.multinomial_naive_bayes import MultinomialNaiveBayes`

[^3]: `from cgpos.models.multinomial_naive_bayes import StupidBayes`

[^4]:Therefore, while most of English follows only [eight inflections](https://en.wikipedia.org/wiki/Inflection#Examples_in_English), a single Greek verb can have [hundreds of word endings](https://en.wiktionary.org/wiki/%CE%BB%CF%8D%CF%89#Inflection) to express every combination of person, number, mood, aspect, voice.


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


