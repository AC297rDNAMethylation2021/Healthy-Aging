##Summary of files

These are pickled sklearn models of type:

Ridge, lasso, linear, kNN, and XGboost regression.

They can be unpicked as follows:

	import pickle
	with open(‘wb_mod_XG_1000’ ,  ‘rb’) as fp:
		wb_mod_XG_1000 = pickle.load(fp)


The numbers 100 and 1000 refer to how many of the top ranked blood CpGs were used in the models

Gen —refers to models with sex as a feature

Stats —refers to models that used the CpG ranking from the Stats modeling feature selection, all other used XBboost  feature selection