`cgpos`: Classical Greek Part-of-Speech Tagging 
==============================
I built an NLP model that does Part-of-Speech Tagging for Classical Greek.

Greek is a hard, low-resource language.

Developing a great POS Tagger will require a lot of attention to detail, smart feature engineering, and a thoughtful approach to model selection. 

Let us motivate the task by providing background on the difficulties.

##  English Part-of-Speech Tagging is kind of easy 

English makes it easy to figure out part-of-speech. 

First off, sentences typically follow a strict Subject-Verb-Object order:

![img/SVO.png](img/SVO.png)

Also, word endings are pretty simple:

![img/conj.png](img/conj.png)

And lastly, we have *a lot* of it:

![img.png](img/eng_tokens.png)

## Greek is hard

Greek, on the other hand, is not so simple.

Firstly, the words can appear in any order (and they do). 

Secondly, part-of-speech is determined by an extremely complex system of word endings:
![img/greek.png](img/greek.png)
*Present tense conjugation one verb φύω, to appear. There are 5 other tenses*

Lastly, we do not have a lot of Greek, and we do have is largely the most difficult and abstruse subset of the language: namely, poetry, philosophical dialogue, and religious texts.

## POS Classical Greek: a true test of NLP 

All of this makes Classical Greek a perfect language to study NLP -- and vice-versa.

# Steps
## Previous work shows 

## Treebanks 

## 

## Tools used in this project
* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management - [article](https://mathdatasimplified.com/2023/06/12/poetry-a-better-way-to-manage-python-dependencies/)
* [hydra](https://hydra.cc/): Manage configuration files - [article](https://mathdatasimplified.com/2023/05/25/stop-hard-coding-in-a-data-science-project-use-configuration-files-instead/)
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [DVC](https://dvc.org/): Data version control - [article](https://mathdatasimplified.com/2023/02/20/introduction-to-dvc-data-version-control-tool-for-machine-learning-projects-2/)
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project

## Set up the environment
1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:
```bash
make env 
```

## Install dependencies
To install all dependencies for this project, run:
```bash
poetry install
```

To install a new package, run:
```bash
poetry add <package-name>
```

## Version your data
To track changes to the "data" directory, type:
```bash
dvc add data
```

This command will create the "data.dvc" file, which contains a unique identifier and the location of the data directory in the file system.

To keep track of the data associated with a particular version, commit the "data.dvc" file to Git:
```bash
git add data.dvc
git commit -m "add data"
```

To push the data to remote storage, type:
```bash
dvc push 
```

## Auto-generate API documentation

To auto-generate API document for your project, run:

```bash
make docs
```
