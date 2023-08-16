Part-of-Speech Tagging for Classical Greek 🏺
==============================
This project builds a fine-trained [part-of-speech tagger](https://en.wikipedia.org/wiki/Part-of-speech_tagging) for Ancient Classical Greek, a [morphologically rich](https://arxiv.org/pdf/2005.01330.pdf) and [low-resource](https://arxiv.org/pdf/2006.07264.pdf) language, using texts from [The Ancient Greek and Latin Dependency Treebank](https://perseusdl.github.io/treebank_data/) and ideas inspired by [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/).

# Model
Ancient Greek is hard to parse due to its use of [word endings](https://en.wiktionary.org/wiki/Appendix:Ancient_Greek_grammar_tables)[^1] to indicate part-of-speech.

This means we cannot necessarily rely on [classical methods that rely heavily on use word order](https://en.wikipedia.org/wiki/Hidden_Markov_model)  to make predictions.


[^1]: This is in contrast to [analytics languages](https://en.wikipedia.org/wiki/Analytic_language) like English that use word order and special words like "had" and "will" to express part-of-speech.

[^2]: Therefore, while most of English follows only [eight inflections](https://en.wikipedia.org/wiki/Inflection#Examples_in_English), a single Greek verb can have [hundreds of word endings](https://en.wiktionary.org/wiki/%CE%BB%CF%8D%CF%89#Inflection) to express every combination of person, number, mood, aspect, voice.


# Run the code
Instructions to run this repository:
## Set environment
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

## Get data
Grab project data:
```
$ cd /dir/to/repository
$ make make_dataset
Initializing data directory
mkdir data/raw/zip
Grabbing Perseus data
...
```

## Run model
Train the part-of-speech tagger using: 
```
$ cd /dir/to/repository
$ make run_model # TODO
Building features
python src/cgpos/features/build_features.py
...
```

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


