
help:
	@cat Makefile

update:
	poetry update
	$(MAKE) test

test:
	poetry install
	poetry run pytest *_test.py
	poetry run black --check *.py
	poetry run mypy .
	poetry run flake8 *.py
	poetry run pylint --jobs=4 *.py
	poetry run isort --check --settings-path=setup.cfg *.py
