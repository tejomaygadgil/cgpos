.DEFAULT_GOAL := help
.PHONY: help clean data lint requirements

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
# 1. PRE-TRAINING DATA
## Get shakespeare dataset
shakespeare: | data/raw/input.txt

data/raw/input.txt:
	wget -P data/raw https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## Process pre-training data.
process_pt_data: | data/interim/pt_beta.pkl data/interim/pt_uni.pkl data/processed/pt_norm.pkl data/processed/pt_syl.pkl

data/interim/pt_beta.pkl data/interim/pt_uni.pkl data/processed/pt_norm.pkl data/processed/pt_syl.pkl: | get_pt_data
	$(info Making pre-training dataset)
	python src/process_data.py pt

## Download pre-training data. (large!)
get_pt_data: | data/raw/diorisis/Achilles\ Tatius\ (0532)\ -\ Leucippe\ and\ Clitophon\ (001).xml

data/raw/diorisis/Achilles\ Tatius\ (0532)\ -\ Leucippe\ and\ Clitophon\ (001).xml: | init_data_dir
	curl -C - -Lo data/zip/diorisis.zip https://figshare.com/ndownloader/files/11296247
	unzip data/zip/diorisis.zip -d data/raw/diorisis

# 2. FINE-TUNING DATA
## Process fine-tuning data.
process_ft_data: | data/reference/ft_targets_map.pkl data/interim/ft_raw.pkl data/interim/ft_clean.pkl data/processed/ft_norm.pkl data/processed/ft_targets.pkl data/processed/ft_syl.pkl

data/reference/ft_targets_map.pkl data/interim/ft_raw.pkl data/interim/ft_clean.pkl data/processed/ft_norm.pkl data/processed/ft_targets.pkl data/processed/ft_syl.pkl: | get_ft_data
	$(info Making fine-tuning dataset)
	python src/process_data.py ft

## Download fine-tuning data.
get_ft_data: | data/raw/treebank_data-master/README.md

data/raw/treebank_data-master/README.md: | init_data_dir
	$(info Grabbing Perseus data)
	curl -C - -Lo data/zip/perseus.zip https://github.com/PerseusDL/treebank_data/archive/master.zip
	unzip data/zip/perseus.zip -d data/raw

# 3. DATA DIRECTORY
## Initialize data directory.
init_data_dir: | data/zip data/raw data/processed data/interim data/reference
	$(info Initializing data directory)

data/zip data/raw data/processed data/interim data/reference:
	mkdir -p $@

## Remove all data in repository.
remove_data: init_data_dir
	$(info Removing data)
	rm -rf data/zip/* data/raw/*  data/processed/* data/interim/* data/reference/*

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################
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
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
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
