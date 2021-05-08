import os
import json

def combine_jupyter_notebooks(notebooks_to_merge, combined_file_name):
    '''
    Combines multiple jupyter notebooks into one
    
    parameters:
    notebooks_to_merge (list): an ordered list of your .ipynb files to merge
    combined_file_name (string): name of the combined .ipynb file which will be generated.
    
    '''
    with open(notebooks_to_merge[0], mode='r', encoding='utf-8') as f:
        a = json.load(f)
    
        for notebook in notebooks_to_merge[1:]:
            with open(notebook, mode='r', encoding='utf-8') as f:
                b = json.load(f)
                a['cells'].extend(b['cells'])
                    # extend here not append so that each dictionary in b['cells']
                    # is added to new dictionary in a['cells']
                    
    with open(combined_file_name, mode='w', encoding='utf-8') as f:
        json.dump(a, f)
