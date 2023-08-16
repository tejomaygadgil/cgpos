.PHONY: clean data lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################



#################################################################################
# SETUP                                                                      #
#################################################################################
## Activate poetry environment
activate_poetry: install_poetry
	poetry shell

## Install poetry environment
install_poetry:
	poetry install
	poetry run pre-commit install

## Run tests
tests:
	pytest

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
## Build features
build_features: | data/reference/feature_map.pkl  data/processed/features.pkl

data/reference/feature_map.pkl  data/processed/features.pkl: | make_dataset
	$(info Building features)
	python src/cgpos/features/build_features.py

## Make Perseus dataset
make_dataset: | data/interim/perseus_raw.pkl data/reference/target_map.pkl data/interim/perseus_normalized.pkl data/processed/cleaned.pkl

data/interim/perseus_raw.pkl data/reference/target_map.pkl data/interim/perseus_normalized.pkl data/processed/cleaned.pkl: | get_data
	$(info Making dataset)
	python src/cgpos/data/make_dataset.py

## Get raw data
get_data: | data/raw/treebank_data-master/README.md data/raw/Greek-Dependency-Trees-master/README.md

data/raw/treebank_data-master/README.md: | data/raw/zip
	$(info Grabbing Perseus data)
	curl -Lo data/raw/zip/perseus.zip https://github.com/PerseusDL/treebank_data/archive/master.zip
	unzip data/raw/zip/perseus.zip -d data/raw

data/raw/Greek-Dependency-Trees-master/README.md: | data/raw/zip
	$(info Grabbing Greek Dependency Trees data)
	curl -Lo data/raw/zip/gorman.zip https://github.com/vgorman1/Greek-Dependency-Trees/archive/master.zip
	unzip data/raw/zip/gorman.zip -d data/raw

data/raw/zip: | init_data_dir
	mkdir $@

## Initialize data directory
init_data_dir: | data/raw data/processed data/interim data/external data/results data/reference
	$(info Initializing data directory)

data/raw data/processed data/interim data/external data/results data/reference:
	mkdir -p $@

## Remove processed data
remove_data: init_data_dir
	$(info Removing project data (excluding data/raw))
	rm -rf data/processed/* data/interim/* data/external/* data/results/* data/reference/*

## Remove all data
remove_all_data: init_data_dir
	$(info Removing all data)
	rm -rf data/raw/*  data/processed/* data/interim/* data/external/* data/results/* data/reference/*

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
