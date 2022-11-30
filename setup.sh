#!/bin/bash


option=$1  # setup option to choose


GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m"


header() {
    echo -e "\n\n${GREEN}=================================================="
    echo -e "$1"
    echo -e "==================================================${NC}\n"
}


install () {
    # install pipx and poetry
    pip3 install --upgrade pip
	pip3 install --user pipx
	pipx ensurepath
    pipx install --python python3 poetry==1.1.15

    # make poetry automatically create a `.venv` for the project.
	poetry config installer.parallel true
	poetry config virtualenvs.create true
	poetry config virtualenvs.in-project true

    # allow poetry to publish on nexus
    poetry config repositories.private https://github.com/pyunc/project-practical-mlops-3-classifier.git

    # install git hook `pre-commit` plugin
    pipx install pre-commit
	pre-commit install

    # Install the project. This will create a .venv on the current repository.
    # Activate the environment with `source .venv/bin/activate` or `poetry shell`.
    # If using `vscode`, point the IDE to use this virtual environment.
	poetry install
}


tests () {
    source .venv/bin/activate

    header "Checking test coverage."
	pytest --cov cookbook --cov-fail-under=100 --cov-report term-missing tests/
}


help () {
    install="To install the project type: \n\t> ./setup.sh install \n"
    tests="\nTo run the tests type: \n\t> ./setup.sh tests \n"
    header "${install}${tests}"
}


case $option in
    install)
        install
        exit;;

    tests)
        tests
        exit;;

    *)
        help
        ;;
esac
