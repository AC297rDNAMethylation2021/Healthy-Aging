##Summary of files

These are pickled sklearn models of type:

Ridge, lasso, linear, and XGboost regression.

They can be unpicked as follows:

	import pickle
	with open(‘breast_mod_100’ ,  ‘rb’) as fp:
		breast_mod_100 = pickle.load(fp)


The numbers 100 and 1000 refer to how many of the top ranked breast CpGs were used in the models

 
