
all:
#	python benchmark.py sgd random
#	python benchmark.py mnb random
	python benchmark.py sgd hyperopt
	python benchmark.py mnb hyperopt
	python benchmark.py sgd skopt
	python benchmark.py mnb skopt

