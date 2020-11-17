
help:
	@cat Makefile

update:
	poetry update
	$(MAKE) test

test:
	poetry install
	poetry run black --check .
	poetry run flake8 pytoolkit
	poetry run mypy pytoolkit
	poetry run pylint --jobs=4 pytoolkit
	poetry run pytest pytoolkit
	poetry run isort --check --settings-path=setup.cfg --recursive pytoolkit
