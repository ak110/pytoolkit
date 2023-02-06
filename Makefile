
help:
	@cat Makefile

update:
	poetry update
	$(MAKE) test

test:
	poetry install --extras=all
	poetry run pyfltr

check:
	poetry run pyfltr --commands=pyupgrade,isort,black,pflake8,mypy
	cd docs && ./update.sh && make html
	CUDA_VISIBLE_DEVICES=none pyfltr --commands=pytest