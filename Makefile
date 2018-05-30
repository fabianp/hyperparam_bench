
all:
#	python newsgroups.py sgd random
#	python newsgroups.py mnb random
	python newsgroups.py sgd hyperopt
	python newsgroups.py mnb hyperopt
	python newsgroups.py sgd skopt
	python newsgroups.py mnb skopt

