##Summary of files

These are pickled python lists of CpG sites. They can be unpicked as follows:

	import pickle
	with open(‘cpgs_XGboost_blood_ranked’ ,  ‘rb’) as fp:
		cpgs_XGboost_blood_ranked = pickle.load(fp)


cpgs\_statsmod\_blood\_ranked

	▪	Top ranked CpG sites in blood by statistical modeling

cpgs\_XGboost\_blood\_ranked

	▪	Top ranked CpG sites in blood by XGboost Importance CV

cpgs\_XGboost\_blood\_ranked\_gender

	▪	Top ranked CpG sites in blood by XGboost Importance CV when sex included as feature

