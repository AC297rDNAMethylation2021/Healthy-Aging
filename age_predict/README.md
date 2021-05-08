# age_predict

Folder structure


	age_predict/
		Combine_jupyter_notebooks.py
			contains function for combining Jupyter notebooks

		data_processing.py
			contains data processing functions

		Loading_EWAS_Aging_Data.py
			contains functions for loading and imputing EWAS data by tissue

		Pickle_unpickle.py
			contains functions to pickle and unpickle objects

		Regression.py
			contains functions to run Linear, XGboost, Ridge, Lasso, kNN regression


	To us this package:

		1) Navigate in a terminal to this folder.

		2) Type: pip install .

		3) Then us as you would a standard python package

				Examples:

					from  age_predict import regression as rg

					from  age_predict import Loading_EWAS_Aging_Data as le

					