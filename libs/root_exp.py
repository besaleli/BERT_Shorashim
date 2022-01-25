from libs.get_docs import getDocs
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import random
import time

cos_sim = torch.nn.CosineSimilarity()
LOGDIR = 'logs/shoresh_distances.txt'


def log_rec(id1, id2, rt1, rt2, f1, f2, r1, r2, similarity):
    with open(LOGDIR, 'a') as f:
        f.write('IDS: {}|{}'.format(str(id1), str(id2)))
        f.write('RAW TOKS: {}|{}'.format(rt1, rt2))
        f.write('FORMS: {}|{}'.format(f1, f2))
        f.write('ROOTS: {}|{}'.format(r1, r2))
        f.write('SIMILARITIES: ' + '|'.join([str(i) for i in similarity.tolist()]))
        f.close()


class RootInstance:
    def __init__(self, instance_id, raw_tok, form, root, embedding, tokenizer_index):
        self.instance_id = instance_id
        self.raw_tok = raw_tok
        self.form = form
        self.root = root
        self.embedding = embedding
        self.tokenizer_index = tokenizer_index
        
    def __eq__(self, other):
        return self.instance_id == other.instance_id

def get_estimated_stats(docs):
    total = 0
    bad_index = 0
    good = 0
    for doc in tqdm(docs):
        for pg in doc:
            for sent in pg:
                for tok in sent:
                    for lem in tok:
                        if lem.pos_tag in ['BN', 'BNT', 'VB'] and lem.shoresh is not None:
                            total += 1
                            if tok.tokenizer_index != 1:
                                bad_index += 1
                            else:
                                good += 1
    
    print('Total verbs: {}'.format(total))
    print('Total unusable (bad index): {}'.format(bad_index))
    print('Ratio: {}%'.format(np.round(good/total*100, 2)))
                                
def get_instances(docs, device=None):
    # get raw data
    currID = 0
    instances = []
    for doc in tqdm(docs):
        for pg in doc:
            for sent in pg:
                for tok in sent:
                    for lem in tok:
                        if lem.pos_tag in ['BN', 'BNT', 'VB'] and lem.shoresh is not None:
                            # retrieve embedding
                            if device is None:
                                embedding = tok.embedding
                            else:
                                embedding = torch.Tensor(tok.embedding, device=device)

                            # make instance
                            instance = RootInstance(currID, tok.raw_tok, 
                                                    lem.form, lem.shoresh, embedding, 
                                                    tok.tokenizer_index)

                            # append to instances and increase currID by 1
                            instances.append(instance)
                            currID += 1
                            
    return instances


def generate_valid_pairs(instances):
    created_pairs = []
    print('getting all possible instance pairs...')
    for i in instances:
        i_id = i.instance_id
        for j in instances:
            j_id = j.instance_id
            if (j_id, i_id) not in created_pairs:
                created_pairs.append((i_id, j_id))
                yield i, j


def get_distance_metrics(instances, sample=None):
    if sample is None:
        usable_instances = instances
    else:
        print('sampling...')
        roots_raw = [i.root for i in instances]
        roots = set([i for i in roots_raw if roots_raw.count(i) > 1])
        roots_sample = random.sample(roots, sample)
        usable_instances = [i for i in instances if i.root in roots_sample]
        print('Total instances to be used: {}'.format(str(len(usable_instances))))
        print('Estimated # Computations: {}'.format(str(((len(usable_instances)**2)-len(usable_instances))/2)))
    
    df_cats = ['id_1', 'id_2', 'raw_tok_1', 'raw_tok_2', 'form_1', 'form_2', 'root_1', 'root_2', 'tok_index_1', 'tok_index_2']
    dfdict = {cat: [] for cat in df_cats}
    
    # make similarity cats
    for l in range(12):
        dfdict['sim_' + str(l)] = []
    
    valid_pairs = generate_valid_pairs(usable_instances)
    
    print('getting metrics...')
    sleepcount = 0
    for i, j in tqdm(valid_pairs):
        
        dfdict['id_1'].append(i.instance_id)
        dfdict['id_2'].append(j.instance_id)

        dfdict['raw_tok_1'].append(i.raw_tok)
        dfdict['raw_tok_2'].append(j.raw_tok)

        dfdict['form_1'].append(i.form)
        dfdict['form_2'].append(j.form)

        dfdict['root_1'].append(i.root)
        dfdict['root_2'].append(j.root)
        
        dfdict['tok_index_1'].append(i.tokenizer_index)
        dfdict['tok_index_2'].append(j.tokenizer_index)

        similarity = cos_sim(i.embedding, j.embedding)
        
        for l in range(12):
            dfdict['sim_' + str(l)].append(similarity[l].item())
            
        log_rec(i.instance_id, j.instance_id, i.raw_tok, j.raw_tok,
               i.form, j.form, i.root, j.root, similarity)
        
        sleepcount += 1
        # sleep for 30sec every 50000 iterations so there's minimal thermal throttling
        if sleepcount % 50000 == 0:
            print('sleeping...')
            time.sleep(30)
            
    return pd.DataFrame(dfdict)