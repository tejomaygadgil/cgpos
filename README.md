Part-of-Speech Tagging for Classical Greek 🏺
==============================
This project builds a fine-trained [part-of-speech tagger](https://en.wikipedia.org/wiki/Part-of-speech_tagging) for Ancient Classical Greek, a [morphologically rich](https://arxiv.org/pdf/2005.01330.pdf) and [low-resource](https://arxiv.org/pdf/2006.07264.pdf) language. We use texts from [The Ancient Greek and Latin Dependency Treebank](https://perseusdl.github.io/treebank_data/) and ideas inspired by [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/).

# Model

# Results


# Code
Instructions to run this repository:
## Get environment
Packages are managed by setting up a `poetry` environment:
1. Install [poetry](https://python-poetry.org/docs/#installation)
2. In the terminal, run:
```
$ git clone https://github.com/tejomaygadgil/cgpos.git
$ cd cgpos/
$ make activate_poetry
```

This command will create and activate a `poetry` instance with the correct packages to run the contents of this repository.  

From then on you can build various parts of the project using `make`.

Generate a convenient helpfile of all available options by running:
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

## Get data
Grab project data:
```
$ cd /dir/to/repository
$ make make_dataset
```

Data will automatically be downloaded to `cgpos/data/`.

It can be removed at any time using these commands: 
```
$ make remove_data  # Excludes raw data
$ make remove_all_data 
```

## Running the model
Build project features using this command:
```
$ cd /dir/to/repository
$ make build_features
```

Run the model using: 
```
$ cd /dir/to/repository
$ make run_model # TODO
```

## Tools
This repository is managed using these tools:
* [`make`](https://www.gnu.org/software/make/) for running code
* [`poetry`](https://python-poetry.org) for package management 
* [`hydra`](https://hydra.cc/) for code reproducibility
* [`black`](https://github.com/psf/black) and [`ruff`](https://github.com/charliermarsh/ruff-pre-commit) for code review


