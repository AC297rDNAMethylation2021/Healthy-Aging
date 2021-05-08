## Summary of files

This is a pickled python list of CpG sites. It can be unpicked as follows:

	import pickle
	with open(‘cpgs_XGboost_brain_ranked’ ,  ‘rb’) as fp:
		cpgs_XGboost_brain_ranked = pickle.load(fp)


cpgs\_XGboost\_brain_ranked

	▪	Top ranked CpG sites in brain by XGboost Cross validation on importance scores
