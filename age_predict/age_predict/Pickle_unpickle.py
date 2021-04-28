import pickle

def get_pickled_object(file):
    with open(file , 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def pickle_object(obj, file_name):
    with open(file_name, 'wb') as fp:
        file_out = pickle.dump(obj, fp)
    print(f'pickled as {file_name}')