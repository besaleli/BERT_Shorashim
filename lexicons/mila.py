import pandas as pd
from tqdm.auto import tqdm

def formatBinyan(b):
    newText = ''
    for char in b:
        if char != "'":
            newText += char
    
    return newText.upper()

def get_lex(remove_ambiguous_terms=True, verbose=False):
    # get lexicon dictionary and format
    lex_dir = 'lexicons/mila_lex.xlsx'
    vlex = pd.read_excel(lex_dir, sheet_name='verb')
    vlex.sort_values(by='root', inplace=True)

    # format binyanim
    vlex['binyan'] = vlex['binyan'].apply(formatBinyan)
    
    # drop dotted col
    vlex.drop(labels=['dotted', 'id'], axis=1, inplace=True)
    
    # drop duplicates
    vlex.dropna(inplace=True)
    
    if remove_ambiguous_terms:
        ambiguous_undotted = set()

        for lem in tqdm(vlex['undotted'].unique()):
            lex = vlex[vlex['undotted'] == lem]

            if len(lex) > 1:
                if verbose:
                    print('AMBIGUOUS LEMMA: ' + lem)
                    print(lex)
                    print()
                for i in lex['undotted'].unique():
                    ambiguous_undotted.add(i)

        return vlex, ambiguous_undotted

    else:
        return vlex

def get_dict(df, fr, to):
    return {i: j for i, j in zip(df[fr].to_list(), df[to].to_list())}

def get_roots_dict():
    return get_dict(get_lex(), 'undotted', 'root')

def get_binyan_dict():
    return get_dict(get_lex(), 'undotted', 'binyan')