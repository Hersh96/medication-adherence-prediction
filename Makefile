.PHONY: setup train test
setup:
	pip install -r requirements.txt
train:
	PYTHONPATH=src python -m adherence.train
test:
	python -m pytest -q
