# globals

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = grid_world

## ------------------------------------------------------------------
## This Makefile contains utilities you can use while developing this
## project.


.PHONY: environment
environment: ## create environment
	pyenv install -s 3.9.5
	pyenv virtualenv 3.9.5 grid_world
	pyenv local grid_world

.PHONY: requirements
requirements: ## install all requirements
	pip install -Ur requirements.txt 

.PHONY: flake-check
flake-check: ## check PEP-8 and other standards with flake8
	@echo ""
	@echo "\033[33mFlake 8 Standards\033[0m"
	@echo "\033[33m=================\033[0m"
	@echo ""
	@python -m flake8 && echo "\n\n\033[32mSuccess\033[0m\n" || (echo \
	"\n\n\033[31mFailure\033[0m\n\n\033[34mManually fix the offending \
	issues\033[0m\n" && exit 1)

.PHONY: black-check
black-check: ## check Black code style
	@echo ""
	@echo "\033[33mBlack Code Style\033[0m"
	@echo "\033[33m================\033[0m"
	@echo ""
	@python -m black --check --exclude="build/|buck-out/|dist/|_build/\
	|pip/|env/|\.pip/|\.git/|\.hg/|\.mypy_cache/|\.tox/|\.venv/" . \
	&& echo "\n\n\033[32mSuccess\033[0m\n" || (python -m black --diff \
	--exclude="build/|buck-out/|dist/|_build/|pip/|env/|\.pip/|\.git/|\
	\.hg/|\.mypy_cache/|\.tox/|\.venv/" . 2>&1 | grep -v -e reformatted -e done \
	&& echo "\n\033[31mFailure\033[0m\n\n\
	\033[34mRun \"\e[4mmake black\e[24m\" to apply style formatting to your code\
	\033[0m\n" && exit 1)

.PHONY: black
black: ## apply the Black code style to code
	black --exclude="build/|buck-out/|dist/|_build/|pip/|env/|\.pip/|\.git/|\.hg/|\.mypy_cache/|\.tox/|\.venv/" .

.PHONY: init
init: git-init environment requirements ## initialize project

.DEFAULT_GOAL := help
