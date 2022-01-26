# a pickle is a lot faster to load so periodt
from libs.get_docs import getDocs
import pickle
from libs.corp_df import *

# retrieve documents
print('Retrieving Documents...')
cdir = 'corpora/articles_stage2/*.json'
docs = getDocs(cdir, verbose=False)

# save to pickle
print('pickling...')
with open('corpora/articles_stage2.pickle', 'wb') as f:
    pickle.dump(docs, f, protocol=-1)