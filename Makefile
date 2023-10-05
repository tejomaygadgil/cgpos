.PHONY: clean data lint requirements

# Download data for pre-training
get_pt_data: | data/raw/diorisis/Achilles\ Tatius\ (0532)\ -\ Leucippe\ and\ Clitophon\ (001).xml

data/raw/diorisis/Achilles\ Tatius\ (0532)\ -\ Leucippe\ and\ Clitophon\ (001).xml: | data/raw/zip
	curl -Lo data/raw/zip/diorisis.zip https://figshare.com/ndownloader/files/11296247
	unzip data/raw/zip/diorisis.zip -d data/raw/diorisis

## Download data for fine-tuning
get_ft_data: | data/raw/treebank_data-master/README.md

data/raw/treebank_data-master/README.md: | data/raw/zip
	$(info Grabbing Perseus data)
	curl -Lo data/raw/zip/perseus.zip https://github.com/PerseusDL/treebank_data/archive/master.zip
	unzip data/raw/zip/perseus.zip -d data/raw

## Initialize data directory
init_data_dir: | data/raw data/processed data/interim data/reference
	$(info Initializing data directory)

data/raw data/processed data/interim data/reference:
	mkdir -p $@

## Remove data
remove_all_data: init_data_dir
	$(info Removing data)
	rm -rf data/raw/*  data/processed/* data/interim/* data/reference/*

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
