
help:
	@cat Makefile

update:
	poetry update
	$(MAKE) test

test:
	poetry install --extras=all
	poetry run pyfltr
	$(MAKE) doc

doc:
	cd docs && ./update.sh && make html
