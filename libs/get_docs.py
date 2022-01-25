import json
import os
import sys
from libs.corp_df import *
from tqdm.auto import tqdm
import glob
import pickle
import time
import numpy as np

FILEDIR = 'corpora/articles_stage2/*.json'
docs = []

def getDocs(corpDir, verbose=False):
    docs = []
    mb = lambda i: np.round(i / (1024**2), 2)
    for name in tqdm(glob.glob(corpDir)):
        doc = doc_from_json(name)
        docs.append(doc)
        if verbose:
            print('First Sentence: ')
            print(doc[0][0])
            print('JSON SIZE: ' + str(os.path.getsize(name)) + 'mb')
            print('OBJ SIZE: ' + str(sys.getsizeof(doc)) + 'mb')
            print()
        
    return docs

def get_pickled_docs(corpDir, verbose=False):
    start = time.time()
    with open(corpDir, 'rb') as f:
        docs = pickle.load(f)
        f.close()
        
    end = time.time()
    
    if verbose:
        print('TIME ELAPSED: {}s'.format(np.round(float(end-start),2)))
        
    return docs