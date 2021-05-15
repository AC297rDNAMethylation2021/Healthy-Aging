# Unhealthy brain analysis

This directory contains the analysis for the unhealthy cohort including applying linear models to the unhealthy cohort and classification.

Directory structure:

      brain/
            read_all_unhealthy_brain_data.ipynb: script used to generate alz_brain_unhealthy_all.csv and hunt_brain_unhealthy_all.csv
            
            produce_combined_healthy_unhealthy.ipynb: used to produce 
            
                              combined_cpgs: list of top 100 CpG sites from HC, Alzheimer's and Huntington's cohorts
                              
                              ombined_healthy_unhealthy_107_cpgs.csv: combined df with top 100 CpG sites from HC, Alzheimer's and Huntington's cohorts
                              
                              ombined_cpgs_healthy_and_alz: list of top 100 CpG sites from HC and Alzheimer's cohorts

            alzheimers/

            classification/

            healthy/

            huntingtons/

            linear_model_comparison/

            models/

            train_test_ids/
            
                              test_train_split_alzheimers_and_huntingtons.ipynb: script to generate working/held out data splits for unhealthy cohorts
                              
                              6 files containing the working and held out data IDs. 2 for each of healthy, Alzheimer's and Huntington's
                              
      
