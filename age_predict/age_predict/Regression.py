from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Run kNN Regression model and plot and print results
from sklearn.neighbors import KNeighborsRegressor
def kNN_regress(X_train, y_train, X_test, y_test, plot=True, ks=[1, 2, 3, 5, 10, 15, 20, 30, 50],cv=5):
    
    # find best k 
    means = []
    for k in ks:
        mod = KNeighborsRegressor(n_neighbors=k)
        scores = cross_val_score(mod, X_train, y_train, cv=cv)
        means.append(scores.mean())
        #print(k, scores.mean())
    k_best = ks[np.argmax(means)]
    print(f'Best K = {k_best}')
    
    #Build fit model
    mod = KNeighborsRegressor(n_neighbors=k_best)
    mod.fit(X_train, y_train)
    
    # Make predictions and evaluate
    preds_train = mod.predict(X_train)
    preds_test = mod.predict(X_test)
    rms_train = (mean_squared_error(y_train, preds_train))**0.5
    rms_test = (mean_squared_error(y_test, preds_test))**0.5
    r2_train = r2_score(y_train, preds_train)
    r2_test = r2_score(y_test, preds_test)
    mae_train = mean_absolute_error(y_train, preds_train)
    mae_test = mean_absolute_error(y_test, preds_test)
    
    # Plot progress over epochs and final true vs predicted age
    if plot:
        fig, ax = plt.subplots(1,2, figsize=(16,4))
        ax[0].scatter(y_train, preds_train, alpha=0.5)
        ax[0].plot(range(20,100), range(20,100), c='red')
        ax[0].set_xlabel('True Age')
        ax[0].set_ylabel('Predicted Age')
        ax[0].grid(True, lw=1.5, ls='--', alpha=0.75)
        ax[0].set_title(f'kNN Regression on training data, k = {k_best}')

        ax[1].scatter(y_test, preds_test, alpha=0.5)
        ax[1].plot(range(20,100), range(20,100), c='red')
        ax[1].set_xlabel('True Age')
        ax[1].set_ylabel('Predicted Age')
        ax[1].grid(True, lw=1.5, ls='--', alpha=0.75)
        ax[1].set_title(f'kNN Regression on testing data, k = {k_best}')
        plt.show()
    
    # print metric
    print(f'The rms on the training data is {rms_train:.3f} years')
    print(f'The rms on the testing data is {rms_test:.3f} years')
    print(f'The r^2 on the training data is {r2_train:.3f}')
    print(f'The r^2 on the testing data is {r2_test:.3f}')
    print(f'The MAe on the training data is {mae_train:.3f} years')
    print(f'The MAE on the testing data is {mae_test:.3f}')
    
    return mod, rms_train, rms_test, r2_train, r2_test

# Run Linear Regression model,  cpgs-data vs age, and plot and print results
from sklearn.linear_model import LinearRegression
def linear_regress(X_train, y_train, X_validation, y_validation, plot=True):

    # Build fit model
    mod = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs='None')
    mod.fit(X_train, y_train)

    # Make predictions and evaluate
    preds_train = mod.predict(X_train)
    preds_validation = mod.predict(X_validation)
    rms_train = (mean_squared_error(y_train, preds_train) )**0.5
    rms_validation = (mean_squared_error(y_validation, preds_validation) )**0.5
    r2_train = r2_score(y_train, preds_train)
    r2_validation = r2_score(y_validation, preds_validation)
    mae_train = mean_absolute_error(y_train, preds_train)
    mae_validation = mean_absolute_error(y_validation, preds_validation)

    # Plot progress over epochs and final true vs predicted age
    if plot:
        fig, ax = plt.subplots(1 ,2, figsize=(16 ,4))
        ax[0].scatter(y_train, preds_train, alpha=0.5)
        ax[0].plot(range(20 ,100), range(20 ,100), c='red')
        ax[0].set_xlim(0, 120)
        ax[0].set_ylim(0, 120)
        ax[0].set_xlabel('True Age')
        ax[0].set_ylabel('Predicted Age')
        ax[0].grid(True, lw=1.5, ls='--', alpha=0.75)
        ax[0].set_title('Linear Regression on training data')

        ax[1].scatter(y_validation, preds_validation, alpha=0.5)
        ax[1].plot(range(20 ,100), range(20 ,100), c='red')
        ax[1].set_xlim(0, 120)
        ax[1].set_ylim(0, 120)
        ax[1].set_xlabel('True Age')
        ax[1].set_ylabel('Predicted Age')
        ax[1].grid(True, lw=1.5, ls='--', alpha=0.75)
        ax[1].set_title('Linear Regression on Validation data')
        plt.show()

    # print metric
    print(f'The rms on the training data is {rms_train:.3f} years')
    print(f'The rms on the validation data is {rms_validation:.3f} years')
    print(f'The r^2 on the training data is {r2_train:.3f}')
    print(f'The r^2 on the validation data is {r2_validation:.3f}')
    print(f'The MAe on the training data is {mae_train:.3f} years')
    print(f'The MAE on the validation data is {mae_validation:.3f}')
    return mod, rms_train, rms_validation, r2_train, r2_validation


# Function for running XGboost regression, cpgs-data vs age, and print results
from xgboost import XGBRegressor
def xgboost_regress(X_train, y_train, X_validation, y_validation, early_stopping_rounds=None, plot=True):
    # Build and fit model
    XG = XGBRegressor(objective='reg:squarederror',
                      n_estimators=200,
                      min_child_weight=1,
                      max_depth=3,
                      subsample=0.7,
                      colsample_bytree=0.5,
                      learning_rate=0.1)

    eval_set = [(X_train, y_train), (X_validation, y_validation)]
    XG.fit(X_train, y_train, eval_metric="rmse", early_stopping_rounds=early_stopping_rounds, eval_set=eval_set,
           verbose=False)

    # Make predictions and evaluate
    preds_train = XG.predict(X_train)
    preds_validation = XG.predict(X_validation)
    rms_train = (mean_squared_error(y_train, preds_train))**0.5
    rms_validation = (mean_squared_error(y_validation, preds_validation))**0.5
    r2_train = r2_score(y_train, preds_train)
    r2_validation = r2_score(y_validation, preds_validation)
    mae_train = mean_absolute_error(y_train, preds_train)
    mae_validation = mean_absolute_error(y_validation, preds_validation)
    results = XG.evals_result()
    epochs = len(results['validation_0']['rmse'])

    # Plot progress over epochs and final true vs predicted age
    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(16, 3.5))
        ax[0].scatter(y_train, preds_train, alpha=0.5)
        ax[0].plot(range(20, 100), range(20, 100), c='red')
        ax[0].set_xlim(0, 120)
        ax[0].set_ylim(0, 120)
        ax[0].set_xlabel('True Age')
        ax[0].set_ylabel('Predicted Age')
        ax[0].grid(True, lw=1.5, ls='--', alpha=0.75)
        ax[0].set_title('XGboost on training data')

        ax[1].scatter(y_validation, preds_validation, alpha=0.5)
        ax[1].plot(range(20, 100), range(20, 100), c='red')
        ax[1].set_xlim(0, 120)
        ax[1].set_ylim(0, 120)
        ax[1].set_xlabel('True Age')
        ax[1].set_ylabel('Predicted Age')
        ax[1].grid(True, lw=1.5, ls='--', alpha=0.75)
        ax[1].set_title('XGboost on Validation data')

        x_axis = range(0, epochs)
        ax[2].plot(x_axis, results['validation_0']['rmse'], label='Train')
        ax[2].plot(x_axis, results['validation_1']['rmse'], label='validation')
        ax[2].legend()
        ax[2].set_ylabel('rms')
        ax[2].set_xlabel('epoch')
        ax[2].set_title('XGBoost rms')
        plt.show()

    # print metrics
    print(f'The number of training epochs was {epochs}')
    print(f'The rms on the training data is {rms_train:.3f} years')
    print(f'The rms on the validation data is {rms_validation:.3f} years')
    print(f'The r^2 on the training data is {r2_train:.3f}')
    print(f'The r^2 on the validation data is {r2_validation:.3f}')
    print(f'The MAE on the training data is {mae_train:.3f} years')
    print(f'The MAE on the validation data is {mae_validation:.3f} years')
    return XG, rms_train, rms_validation, r2_train, r2_validation, XG.feature_importances_


# Function for running a ridge Regression model with CV on alpha, cpgs-data vs age,
# and plotting and printing th results
from sklearn.linear_model import RidgeCV
def ridgeCV_regress(X_train, y_train, X_validation, y_validation, plot=True,
                    alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5], cv=5):
    # Build fit model
    mod = mod = RidgeCV(alphas=alphas, cv=cv)
    mod.fit(X_train, y_train)

    # Make predictions and evaluate
    preds_train = mod.predict(X_train)
    preds_validation = mod.predict(X_validation)
    rms_train = (mean_squared_error(y_train, preds_train))**0.5
    rms_validation = (mean_squared_error(y_validation, preds_validation))**0.5
    r2_train = r2_score(y_train, preds_train)
    r2_validation = r2_score(y_validation, preds_validation)
    mae_train = mean_absolute_error(y_train, preds_train)
    mae_validation = mean_absolute_error(y_validation, preds_validation)

    # Plot final true vs predicted age
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        ax[0].scatter(y_train, preds_train, alpha=0.5)
        ax[0].plot(range(20, 100), range(20, 100), c='red')
        ax[0].set_xlim(0, 120)
        ax[0].set_ylim(0, 120)
        ax[0].set_xlabel('True Age')
        ax[0].set_ylabel('Predicted Age')
        ax[0].grid(True, lw=1.5, ls='--', alpha=0.75)
        ax[0].set_title('Ridge Regression on training data')

        ax[1].scatter(y_validation, preds_validation, alpha=0.5)
        ax[1].plot(range(20, 100), range(20, 100), c='red')
        ax[1].set_xlim(0, 120)
        ax[1].set_ylim(0, 120)
        ax[1].set_xlabel('True Age')
        ax[1].set_ylabel('Predicted Age')
        ax[1].grid(True, lw=1.5, ls='--', alpha=0.75)
        ax[1].set_title('Ridge Regression on Validation data')
        plt.show()

    # print metric
    print(f'The rms on the training data is {rms_train:.3f} years')
    print(f'The rms on the validation data is {rms_validation:.3f} years')
    print(f'The r^2 on the training data is {r2_train:.3f}')
    print(f'The r^2 on the validation data is {r2_validation:.3f}')
    print(f'The MAE on the training data is {mae_train:.3f} years')
    print(f'The MAE on the validation data is {mae_validation:.3f} years')
    print(f'Optimal alpha from CV = {mod.alpha_}')
    return mod, rms_train, rms_validation, r2_train, r2_validation


# Function for running a lasso Regression model with CV on alpha, cpgs-data vs age,
# and plotting and printing the results
from sklearn.linear_model import LassoCV
def lassoCV_regress(X_train, y_train, X_validation, y_validation, plot=True,
                    alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5], cv=5):
    # Build fit model
    mod = mod = LassoCV(alphas=alphas, cv=cv)
    mod.fit(X_train, y_train)

    # Make predictions and evaluate
    preds_train = mod.predict(X_train)
    preds_validation = mod.predict(X_validation)
    rms_train = (mean_squared_error(y_train, preds_train))**0.5
    rms_validation = (mean_squared_error(y_validation, preds_validation))**0.5
    r2_train = r2_score(y_train, preds_train)
    r2_validation = r2_score(y_validation, preds_validation)
    mae_train = mean_absolute_error(y_train, preds_train)
    mae_validation = mean_absolute_error(y_validation, preds_validation)

    # Plot final true vs predicted age
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        ax[0].scatter(y_train, preds_train, alpha=0.5)
        ax[0].plot(range(20, 100), range(20, 100), c='red')
        ax[0].set_xlim(0, 120)
        ax[0].set_ylim(0, 120)
        ax[0].set_xlabel('True Age')
        ax[0].set_ylabel('Predicted Age')
        ax[0].grid(True, lw=1.5, ls='--', alpha=0.75)
        ax[0].set_title('Lasso Regression on training data')

        ax[1].scatter(y_validation, preds_validation, alpha=0.5)
        ax[1].plot(range(20, 100), range(20, 100), c='red')
        ax[1].set_xlim(0, 120)
        ax[1].set_ylim(0, 120)
        ax[1].set_xlabel('True Age')
        ax[1].set_ylabel('Predicted Age')
        ax[1].grid(True, lw=1.5, ls='--', alpha=0.75)
        ax[1].set_title('Lasso Regression on Validation data')
        plt.show()

    # print metric
    print(f'The rms on the training data is {rms_train:.3f} years')
    print(f'The rms on the validation data is {rms_validation:.3f} years')
    print(f'The r^2 on the training data is {r2_train:.3f}')
    print(f'The r^2 on the validation data is {r2_validation:.3f}')
    print(f'The MAE on the training data is {mae_train:.3f} years')
    print(f'The MAE on the validation data is {mae_validation:.3f} years')
    print(f'Optimal alpha from CV = {mod.alpha_}')
    return mod, rms_train, rms_validation, r2_train, r2_validation


# repeat_XGBoost numtrials times using a different train/val split each time
# returns importance scores for all cpgs for each run in a dataframe
def repeat_XGBoost(df_processed, numtrials, early_stopping_rounds=10, test_size=0.20):
    # Assumes df_processed is a dataframe with the first two columns ['tissue', 'age']
    # and the cpgs, with each value a DNA methylation beta value 
    X = df_processed.iloc[:, 2:]
    y = df_processed.age
    importances = []
    rms_train = []
    rms_validation = []
    r2_train = []
    r2_validation = []
    # Fir XGboost model numtrials times
    for i in range(numtrials):
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=test_size, random_state=(i+1))

        # Fit model
        # Hyperparamters used were determined separately by cross val
        XG = XGBRegressor(objective='reg:squarederror',
                          n_estimators=200,
                          min_child_weight=1,
                          max_depth=3,
                          subsample=0.7,
                          colsample_bytree=0.5,
                          learning_rate=0.1)

        eval_set = [(X_train, y_train), (X_validation, y_validation)]
        XG.fit(X_train, y_train, eval_metric="rmse", early_stopping_rounds=early_stopping_rounds, eval_set=eval_set,
               verbose=False)

        # Make predictions and print r^2 on validation set
        preds_train = XG.predict(X_train)
        preds_validation = XG.predict(X_validation)
        rms_train.append((mean_squared_error(y_train, preds_train))**0.5)
        rms_validation.append((mean_squared_error(y_validation, preds_validation))**0.5)
        r2_train.append(r2_score(y_train, preds_train))
        r2_validation.append(r2_score(y_validation, preds_validation))
        print(i + 1, f'r^2 validation = {r2_score(y_validation, preds_validation)}')
        importances.append(XG.feature_importances_)

    # Make a dataframes with importance scores from each trial as columns and the trials as rows
    df_imp = pd.DataFrame(importances, columns=df_processed.columns[2:])
    df_imp = df_imp.transpose()
    cols = []
    for i in range(1, numtrials + 1):
        cols.append('trial_' + str(i))
    df_imp.columns = cols
    df_imp['Mean'] = df_imp.mean(axis=1)
    df_imp['Std'] = df_imp.std(axis=1)
    return df_imp, rms_train, rms_validation, r2_train, r2_validation


# Takes df of importance scores produced by repeat_XGBoost and sorts by mean importance score
def importances_sorted_by_mean(df_imp):
    df_imp_sorted = df_imp.sort_values('Mean', ascending=False)
    return df_imp_sorted


# Makes a histogram of the frequecy of cpgs in the top (top_num) importance scores
def histogram_of_top_CpGs_by_importance(df_imp, top_num=100):
    vs = []
    inds = []
    # Sort each column of df_imp by importance score and take their top cpgs into inds 
    # and their scores into vs
    for col in df_imp.columns[:-2]:
        c = df_imp[col].sort_values(ascending=False)
        vs.append(c[:top_num])
        inds = inds + list(c.index[:top_num])
    h = pd.Series(inds).value_counts()
    # Plotting the first 100 importance scores
    plt.figure(figsize=(30, 12))
    plt.bar(h.index[:100], h[:100])
    plt.title('Frequency of CpGs in the top 100 importances')
    plt.ylabel('Frequency')
    plt.xlabel('CpG')
    plt.xticks(fontsize=16, rotation=90)
    plt.show()
    return inds, vs, h

# Function for testing a model on test data, no fitting,
# and plotting and printing the results
def test_model_on_heldout_data(X_saved, y_saved, model, mtype='Linear Regression', figsize=(8, 4)):
    # Using  model to make predictions on saved data
    preds_on_saved = model.predict(X_saved)
    MSE_validation = mean_squared_error(y_saved, preds_on_saved)
    rms_validation = (mean_squared_error(y_saved, preds_on_saved)) ** 0.5
    r2_validation = r2_score(y_saved, preds_on_saved)
    MAE_validation = mean_absolute_error(y_saved, preds_on_saved)
    r_validation_corr = np.corrcoef(y_saved, preds_on_saved)[0, 1]  # preds_on_saved

    # Plotting results
    model_type = mtype

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(y_saved, preds_on_saved, alpha=0.5)
    ax.plot(range(20, 100), range(20, 100), c='red')
    ax.set_xlabel('True Age')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 120)
    ax.set_ylabel('Predicted Age')
    ax.grid(True, lw=1.5, ls='--', alpha=0.75)
    ax.set_title(f'{model_type} on test data', fontsize=(18))
    plt.show()

    # print metric
    print(f'The MSE on Test data is {MSE_validation:.3f} years')
    print(f'The rms on Test data is {rms_validation:.3f} years')
    print(f'The r^2 on Test data is {r2_validation:.3f}')
    print(f'The mae on Test data is {MAE_validation:.3f} years')
    print(f'The correlation on the Test data true-predicted is {r_validation_corr:.3f}')

    return preds_on_saved, MSE_validation, rms_validation, r2_validation, MAE_validation, r_validation_corr

# given two list of cpgs, returns a list that the two list have in common
def count_common_cpgs(cpgs1, cpgs2, verbose=True):
    if verbose == True:
        print('Common cpgs')
    count = 0
    common_list = []
    for i, cpg in enumerate(cpgs1):
        if cpg in cpgs2:
            common_list.append(cpg)
            if verbose==True:
                print(i, cpg)
            count += 1
        else:
            pass
    print(f'\n{count} cpgs in common')
    return common_list
