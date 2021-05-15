# Unhealthy brain analysis

This directory contains the analysis for the unhealthy cohort including applying linear models to the unhealthy cohort and classification.

Directory structure:

      brain/
            read_all_unhealthy_brain_data.ipynb: script used to generate alz_brain_unhealthy_all.csv and hunt_brain_unhealthy_all.csv
            
            produce_combined_healthy_unhealthy.ipynb: script used to produce 
            
                              combined_cpgs: list of top 100 CpG sites from HC, Alzheimer's and Huntington's cohorts
                              
                              combined_healthy_unhealthy_107_cpgs.csv: combined df with top 100 CpG sites from HC, Alzheimer's and Huntington's cohorts
                              
                              combined_cpgs_healthy_and_alz: list of top 100 CpG sites from HC and Alzheimer's cohorts
                              
            get_shared_healthy_unhealthy.ipynb: script used to produce:
            
                              alz_brain_top_55.csv: Alzheimer's data for top 55 HC CpGs
                              
                              hunt_brain_top_55.csv: Huntingtons's data for top 55 HC CpGs
                              
            combined_cpgs_healthy_and_alz: list of top 100 CpG sites from HC and Alzheimer's cohorts
            
            combined_cpgs: list of top 100 CpG sites from HC, Alzheimer's and Huntington's cohorts

            alzheimers/
            
                              xgb_alz_brain_cpgs: XGBoost ordering of CpG sites most associated with aging in Alzheimers's cohort

            classification/
            
                              classifying_with_healthy_cpgs.ipynb: script used to run classification for:
                              
                                                HC vs Alzheimer's vs Huntington's for top 55 HC CpGs
                                                
                                                HC vs Alzheimer's for top 55 HC CpGs
                                                
                                                HC vs Alzheimer's for residuals from HC model on top 55 HC CpGs
                              
                              classifying_with_unhealthy_cpgs.ipynb: script used to:
                                                
                                                Fit linear models to Huntington's cohort using top Huntington's CpGs
                                                
                                                Apply Huntington's model to HC data
                                                
                                                Fit linear models to Alzheimers's cohort using top Alzheimers's CpGs
                                                
                                                Apply Alzheimers's model to HC data

                                                Fit logistic regression classifier for HC vs Alzheimer's for residuals from Alzheimer's model
                                                
                                                Fit logistic regression classifier for HC vs Alzheimer's for top Alzheimer's CpGs

            healthy/
            
                              cpgs_XGboost_brain_ranked: XGBoost ordering of CpG sites most associated with aging in HC cohort

            huntingtons/
            
                              xgb_hunt_brain_cpgs: XGBoost ordering of CpG sites most associated with aging in Huntington's cohort

            linear_model_comparison/
                              
                              brain_applying_hc_models.ipynb: script to apply HC models to unhealthy cohorts without re-training
                              
                              brain_training_unhealthy_on_healthy_cpgs.ipynb: script to train models on uhealthy cohorts using top healthy CpGs

            models/
            
                              contains linear and XGBoost models from HC analysis

            train_test_ids/
            
                              test_train_split_alzheimers_and_huntingtons.ipynb: script to generate working/held out data splits for unhealthy cohorts
                              
                              6 files containing the working and held out data IDs. 2 for each of healthy, Alzheimer's and Huntington's
                              
      
