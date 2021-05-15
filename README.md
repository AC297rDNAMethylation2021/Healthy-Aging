# Healthy-Aging

Folder structure



	age_predict/
		age_predict/
			contains a custom python package we developed for data analysis
			


	analysis/

		EWAS/ (indicates data was from the Epigenome Wide Association Datahub)
	
			blood/
				contains DNA Methylation analysis with data from whole blood
			
			brain/
				contains DNA Methylation analysis with data from whole blood
			
			breast/
				contains DNA Methylation analysis with data from whole blood
			
			leukocyte/
				contains DNA Methylation analysis with data from whole blood
			
			healthy_vs_unhealthy/
				contains DNA Methylation analysis of healthy vs unhealthy
			
			gene_identification/
				contains mappings of CpG sites to genes
			


		PPMI/ (indicates data used was from the Parkinson's Progression Marker Initiative database)
			
			blood_chem_EDA/
				contains PPMI blood chemistry EDA

			ppmi_120_methylation_profiling/
				contains PPMI methylation data EDA

			data_summary.xlsx
				contains summary of data in various PPMI files

	submissions/

		presentations/
			contains team presentations

		AC297r Merck Statement of Work.pdf
			team statement of work     



## NOTES ON DATA

* The data for this project came from the EWAS Data Hub https://bigd.big.ac.cn/ewas/datahub/index. 

* The data having to do with healthy individuals came from the EWAS-pre-prepared data cut called "age\_methylation\_v1.zip", and its meta data was from the file "sample\_age\_methylation\_v1.zip". When unzipped the data file used was then called "age\_methylation_v1.txt". It is 22 GB in size, too large to place on GitHub. It can be downloaded at: https://bigd.big.ac.cn/ewas/datahub/download

* In the following directory: 'Healthy-Aging/analysis/EWAS/blood/feature\_selection' there is a jupiter notebook entitled "Load\_all\_data\_select\_out\_tissue\_save\_ranked\_dfs". If provided with a path to the data file "age_methylation\_v1.txt" and a tissue type, it will read in the data and select out data from just that tissue. Then, if you like,  it will impute the missing values, using our standard procedure, and then create Train and Test sets containing data from just the cpg sites we have ranked as most important. Then, these datasets are saved as pandas dataframes in the feature_selection directories. It is these dataframes that were then used by the modeling routines in the modeling directories.

* The data on unhealthy individuals came from the EWAS-pre-prepared data cut called "disease\_methylation\_v1.zip" and its metadata was from the file "sample\_disease\_methylation\_v1.zip". This files is too large to place on GitHub. It can be downloaded at: https://bigd.big.ac.cn/ewas/datahub/download

* Cuts of the data required for the analysis can be produced using the following scripts:

alz_brain_top_56.csv and hunt_brain_top_56.csv: get_shared_healthy_unhealthy.ipynb

combined_healthy_unhealthy_107_cpgs.csv: produce_combined_healthy_unhealthy.ipynb

alz_brain_unhealthy_all.csv and hunt_brain_unhealthy_all.csv: read_all_unhealthy_brain_data.ipynb





