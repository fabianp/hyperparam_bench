
all:
	python benchmark.py newsgroups sgd random
	python benchmark.py newsgroups mnb random
	python benchmark.py newsgroups sgd hyperopt
	python benchmark.py newsgroups mnb hyperopt
	python benchmark.py newsgroups sgd skopt
	python benchmark.py newsgroups mnb skopt
	python benchmark.py housing ridge skopt
	python benchmark.py housing ridge hyperopt
	python benchmark.py housing ridge random






