
help:
	@cat Makefile

update:
	poetry update
	$(MAKE) test

test:
	# poetry install --extras "tests tensorflow"
	poetry install --extras "tests"
	poetry run pyfltr pytoolkit
