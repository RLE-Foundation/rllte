SHELL=/bin/bash
LINT_PATHS=rllte/ tests/

pytest: 
	sh ./scripts/run_tests.sh

pytype:
	pytype -j auto ${LINT_PATHS}

type: pytype

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff ${LINT_PATHS} --select=E9,F63,F7,F82 --show-source --fix
	# exit-zero treats all errors as warnings.
	ruff ${LINT_PATHS} --exit-zero

format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black ${LINT_PATHS}

commit-checks: format type lint

build:
	python -m build

twine:
	python -m twine upload --repository pypi dist/*

gendocs:
	gendocs --config docs/mkgendocs.yml
