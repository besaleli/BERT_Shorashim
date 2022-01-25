from libs.corp_df import *
from libs.Embeddings import *
from libs.root_exp import *
from tqdm.auto import tqdm
from libs.get_docs import getDocs
import pandas as pd
import torch

# instantiate cuda device
cuda0 = torch.device('cuda:0')
print('Using the following device:')
print(torch.cuda.get_device_name(0))

# retrieve documents
print('Retrieving Documents...')
cdir = 'corpora/articles_stage2/*.json'
docs = getDocs(cdir, verbose=False)

# get estimated states
get_estimated_stats(docs)

# retrieve instances
instances = get_instances(docs)

# get metrics
metrics = get_distance_metrics(instances, sample=50)

# save to csv
print('Saving CSV...')
metrics.to_csv('data/metrics.csv')