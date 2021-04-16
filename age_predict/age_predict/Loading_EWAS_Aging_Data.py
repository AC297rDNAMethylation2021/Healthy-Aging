import csv
import re
import warnings
import numpy as np
import pandas as pd



def read_write_age_data_by_tissue(file_in, file_out, search_term, num_rows='all', verbose=True):
    '''
    Built to operate on the EWAS aging dataset text file named 'age_methylation_v1.txt'

    Takes an input csv file, a file name to write to, and a search term which should equal a type of tissue
    in the tissue row of the data (3rd row).

    Writes to a csv file the input data but containing only the columns with the designated tissue of interest

    :params
        file_in (string) (file.csv) path to age_methylation_v1.txt
        file_out (string) name of csv file to write
        search_term (string) term to search for in tissue row

    :returns
        None
    '''
    with open(file_in, 'r') as file_in:

        csv_in = csv.reader(file_in ,delimiter= '\t')

        header = next(csv_in)
        age = next(csv_in)
        tissue = next(csv_in)

        indices = [0]
        for i, element in enumerate(tissue):
            if re.search(search_term, element):
                indices.append(i)

        if len(indices) == 0:
            print('Tissue not found')
            return None

        with open(file_out, 'w') as f_out:
            csv_out = csv.writer(f_out)

            new_header =  findElements(header, indices)
            new_age =  findElements(age, indices)
            new_tissue =  findElements(tissue, indices)

            csv_out.writerow(new_header)

            count = 0
            if num_rows=='all':
                for line in csv_in:
                    save = findElements(line, indices)
                    csv_out.writerow(save)
                    count +=1
                    if count % 10000 == 0:
                        print(f'finished line {count}')
            else:
                for line in csv_in:
                    if count >= num_rows:
                        break
                    else:
                        save = findElements(line, indices)
                        csv_out.writerow(save)
                        count +=1
                        if verbose==True:
                            if count % 10000 == 0:
                                print(f'finished line {count}')
        print(f'\n{count} lines sent to file {file_out} with the tissue field containing {search_term}')
        return new_header, new_age, new_tissue

def findElements(lst1, lst2):
    '''
    returns a list of those elements in list1 that are at the position indices given in list2
    Used by read_write_age_data_by_tissue()
    '''
    return [lst1[i] for i in lst2]

def load_EWAS_aging_by_tissue(file_in, file_out, search_term, num_rows='all', verbose=True):
    '''
        Takes as input the EWAS file 'age_methylation_v1.txt' and returns a dataframe containing
        only data of the tissue type specified with the search_term param. Also writes a file of
        the specified data.

        :params
            file_in (string): name of file to read , should be 'age_methylation_v1.txt'
            file_out (string): name of file to write
            search_term (string) term to search for in tissue row
            num_rows (int) number of rows to take from original file
            verbose (boolean) activates some printing during the

        :returns
            df_t (pandas dataframe, rows = sample, columns = cpg sites): transposed version of the input data with
                  only data from the tissue type specified
            This function writes a file like the input but with only the data from the tissue type specified
    '''

    # Step through file_in writing rows where tissue contains search_term to a new file, file_out
    new_header, new_age, new_tissue = read_write_age_data_by_tissue(file_in, file_out, search_term,
                                                                    num_rows=num_rows, verbose=verbose)
    # Reading data back in in chunks
    warnings.simplefilter(action='ignore', category=Warning)
    chunksize = 10000
    dfs = []
    df_chunk = pd.read_csv(file_out, header=0, chunksize=chunksize)
    for chunk in df_chunk:
        dfs.append(chunk)

    # Combining chunks into 1 dataframe
    df = pd.concat(dfs).set_index('sample_id')

    df_t = df.transpose()

    # Inserting columns for tissue and age and setting age dtype to float and then rounding to int
    df_t.insert(0, 'age', new_age[1:])
    df_t.insert(0, 'tissue', new_tissue[1:])
    df_t.age = df_t.age.astype('float')
    df_t.age = df_t.age.apply(np.rint)
    df_t.age = df_t.age.astype('int64')

    return df_t

def splitting_and_imputing(df, input_percent=10, fraction_test=0.25, seed=2021):
    '''
    Takes a dataframe, splits it to Train and Test sets by the specified fraction,
    imputes train with the means of the training columns after throwing away all columns
    with more than input_percent number of NaNs. Then imputes test by keeping the same columns
    as Train and imputing with the means of the training columns.

    :param df: (pandas dataframe) to split
    :param input_percent: (int) if NaNs in a column are more than this percent the column will be discarded
    :param fraction_test: (float) fraction of samples for test set
    :param seed: (int) Seed for random number generator

    :return:
        df_train_imp (pandas dataframe) training df imputed
        df_test_imp (pandas dataframe) tresting df imputed

    '''

    # ---Splitting data into Train and Test by random selection of rows

    num_samples = df.shape[0]
    num_for_saving = int(round(df.shape[0] * fraction_test))
    np.random.seed(seed)
    saved_index = np.random.choice(np.arange(num_samples), size=num_for_saving, replace=False)
    keep_index = []
    for num in range(num_samples):
        if num not in saved_index:
            keep_index.append(num)

    df_test= df.iloc[saved_index, :]
    df_train = df.iloc[keep_index, :]

    # ---Impute Training data---

    # Drop columns with 10% or more NaNs
    df_train_imp = df_train.dropna(thresh=(100 - input_percent)*.01 * df_train.shape[0], axis=1)
    # Impute NaNs in training data with training column means
    train_column_means = df_train_imp.mean()
    df_train_imp = df_train_imp.fillna(train_column_means)

    # ---Impute Testing data---

    # keep only the same columns as in the imputed training data
    df_test_imp = df_test[df_train.columns]
    # impute with the means of the training data columns
    df_test_imp = df_test_imp.fillna(train_column_means)

    return df_train_imp, df_test_imp