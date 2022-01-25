import torch
import json
from libs.Embeddings import *
from libs.Lemmatizer import *
import tqdm

def to_master_json(docs, fileDir=None, indent=4):
    if fileDir is not None:
        with open(fileDir, 'w') as f:
            json.dump([d.to_dict() for d in tqdm(docs)], f, indent=indent)
    else:
        return json.dumps([d.to_dict() for d in tqdm(docs)], indent=indent)
    
def doc_from_master_json(fileDir):
    with open(fileDir, 'r') as f:
        raw_docs = json.load(f)
        f.close()
    
    docs = []
    for doc in raw_docs:
        doc_id = raw_doc['doc_id']
        paragraphs = []

        for raw_p in raw_doc['paragraphs']:
            pg_id = raw_p['pg_id']
            sentences = []

            for raw_sent in raw_p['sentences']:
                sent_id = raw_sent['sent_id']
                tokens = []

                for raw_t in raw_sent['tokens']:
                    tok_id = raw_t['tok_id']
                    raw_tok = raw_t['raw_tok']
                    lemmas = []
                    embedding = torch.FloatTensor(raw_t['embedding'])

                    for raw_lem in raw_t['lemmas']:
                        lemma = raw_lem['lemma']
                        form = raw_lem['form']
                        pos_tag = raw_lem['pos_tag']
                        feats = raw_lem['feats']
                        index = raw_lem['index']
                        binyan = raw_lem['binyan']
                        shoresh = raw_lem['shoresh']

                        lemmas.append(Lemma(lemma, form, pos_tag, feats, 
                                            index, binyan=binyan, shoresh=shoresh))

                    tokens.append(Token(tok_id, raw_tok, lemmas=lemmas, embedding=embedding))

                sentences.append(Sentence(sent_id, tokens))

            paragraphs.append(Paragraph(pg_id, sentences))

        docs.append(Document(doc_id, paragraphs))
    
    return docs

def doc_from_json(fileDir):
    with open(fileDir, 'r') as f:
        raw_doc = json.load(f)
        f.close()
        
    doc_id = raw_doc['doc_id']
    paragraphs = []
    
    for raw_p in raw_doc['paragraphs']:
        pg_id = raw_p['pg_id']
        sentences = []
        
        for raw_sent in raw_p['sentences']:
            sent_id = raw_sent['sent_id']
            tokens = []
            
            for raw_t in raw_sent['tokens']:
                tok_id = raw_t['tok_id']
                raw_tok = raw_t['raw_tok']
                tok_index = raw_t['tokenizer_index']
                lemmas = []
                embedding = torch.FloatTensor(raw_t['embedding'])
                
                for raw_lem in raw_t['lemmas']:
                    lemma = raw_lem['lemma']
                    form = raw_lem['form']
                    pos_tag = raw_lem['pos_tag']
                    feats = raw_lem['feats']
                    index = raw_lem['index']
                    binyan = raw_lem['binyan']
                    shoresh = raw_lem['shoresh']
                    
                    lemmas.append(Lemma(lemma, form, pos_tag, feats, 
                                        index, binyan=binyan, shoresh=shoresh))
                    
                tokens.append(Token(tok_id, raw_tok, lemmas=lemmas, 
                                    embedding=embedding, tokenizer_index=tok_index))
                
            sentences.append(Sentence(sent_id, tokens))
            
        paragraphs.append(Paragraph(pg_id, sentences))
        
    return Document(doc_id, paragraphs)
                    

class Document:
    def __init__(self, doc_id, paragraphs):
        self.doc_id = doc_id
        self.paragraphs = paragraphs
        
    def __iter__(self):
        return iter(self.paragraphs)
    
    def __str__(self):
        return '\n\n'.join([str(i) for i in self.paragraphs])
    
    def __len__(self):
        return len(self.paragraphs)
    
    def __getitem__(self, i):
        return self.paragraphs[i]
    
    def to_dict(self):
        return {'doc_id': self.doc_id, 'paragraphs': [p.to_dict() for p in self.paragraphs]}
    
    def to_json(self, fileDir=None, indent=4):
        docDict = self.to_dict()
        if fileDir is not None:
            with open(fileDir, 'w') as f:
                json.dump(docDict, f, indent=indent)
        else:
            return json.dumps(doctDict, indent=indent)
        
        
class Paragraph:
    def __init__(self, pg_id, sentences):
        self.pg_id = pg_id
        self.sentences = sentences
        
    def __iter__(self):
        return iter(self.sentences)
    
    def __str__(self):
        return '\n'.join([str(i) for i in self.sentences])
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, i):
        return self.sentences[i]
    
    def to_dict(self):
        return {'pg_id': self.pg_id, 'sentences': [s.to_dict() for s in self.sentences]}
        
class Sentence:
    def __init__(self, sent_id, tokens):
        self.sent_id = sent_id
        self.tokens = tokens
        
    def __iter__(self):
        return iter(self.tokens)
    
    def __str__(self):
        return ' '.join([i.raw_tok for i in self.tokens])
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, i):
        return self.tokens[i]
    
    def encode_sent(self, model):
        tokens, indexed_tokens, embeddings = model.encode_sent(self)
        
        for tok_obj, index, sent_tok, embedding in zip(self.tokens, indexed_tokens, tokens, embeddings):
            if tok_obj.raw_tok == sent_tok:
                tok_obj.setEmbedding(embedding[1:])
                tok_obj.tokenizer_index = index
                
    def lemmatize_sent(self):
        # set all lemma data members in all tokens to []
        for tok in self.tokens:
            tok.lemmas = []
        
        # the lemmatizer doesn't like acronyms so replace them with native chars
        toks_cleaned = []
        """for tok in self.tokens:
            if len(tok.raw_tok) > 1:
                if '"' in tok.raw_tok:
                    toks_cleaned.append('ACR')
                elif "'" in tok.raw_tok:
                    toks_cleaned.append('LOANW')
                else:
                    toks_cleaned.append(tok.raw_tok)
            else:
                toks_cleaned.append(tok.raw_tok)"""
        
        for tok in self.tokens:
            rt = tok.raw_tok
            tok_cleaned = ''
            # if len(rt) > 1:
            if '"' in rt or "'" in rt:
                for ch in rt:
                    if ch == '"':
                        tok_cleaned = '״'
                    elif ch == "'":
                        tok_cleaned += '׳'
                    else:
                        tok_cleaned += ch
            else:
                tok_cleaned = rt
                
            # else:
                # tok_cleaned = rt
                
            toks_cleaned.append(tok_cleaned)
            
        # retrieve data from server
        data = Lemmatizer.getData(' '.join(toks_cleaned))
        # parse
        records = [d.split('\t') for d in data.split('\n') if len(d) != 0]
        # make Lemma obj and append to respective token's lemmas data member
        for record in records:
            form = record[2]
            lemma = record[3]
            pos_tag = record[4]
            feats = record[6]
            index = int(record[7])
            
            l = Lemma(lemma, form, pos_tag, feats, index)
            
            try:
                self.tokens[index-1].lemmas.append(l)
            except Exception:
                print('FAILED SENTENCE: ')
                print(str(self))
                print('Record in question: ')
                print(record)
                print('Length of sentence: ')
                print(len(self))
                print()
                print()
                print('Complete record:')
                [print(r) for r in records]
                
    def to_dict(self):
        return {'sent_id': self.sent_id, 'tokens': [t.to_dict() for t in self.tokens]}
        
class Token:
    def __init__(self, tok_id, raw_tok, lemmas=None, embedding=None, tokenizer_index=None):
        self.tok_id = tok_id
        self.raw_tok = raw_tok
        self.lemmas = lemmas if lemmas is not None else []
        self.embedding = embedding if embedding is not None else []
        self.tokenizer_index  = tokenizer_index
            
    def __iter__(self):
        return iter(self.lemmas)
    
    def __str__(self):
        return self.raw_tok
    
    def __len__(self):
        return len(self.lemmas)
    
    def __getitem__(self, i):
        return self.lemmas[i]
    
    def setEmbedding(self, embedding):
        self.embedding = embedding
        
    def getEmbedding(self, layers=None, method='sum'):
        if type(layers) == int:
            return self.embedding[layers]
        elif type(layers) == list:
            selected_layers = [self.embedding[i] for i in layers]
            if method == 'sum':
                return sum(selected_layers)
            elif method == 'stack':
                return torch.stack(selected_layers)
            elif method == 'generator':
                return (i for i in selected_layers)
            else:
                return self.embedding
        else:
            return self.embedding
        
    def to_dict(self):
        return {'tok_id': self.tok_id, 
                'raw_tok': self.raw_tok, 
                'tokenizer_index': self.tokenizer_index,
                'lemmas': [l.to_dict() for l in self.lemmas], 
                'embedding': self.embedding.tolist()}
        
class Lemma:
    def __init__(self, lemma, form, pos_tag, feats, index, binyan=None, shoresh=None):
        self.lemma = lemma
        self.form = form
        self.pos_tag = pos_tag
        self.feats = feats
        self.index = index
        self.binyan = binyan
        self.shoresh = shoresh
        
    def to_dict(self):
        return {'lemma': self.lemma,
                'form': self.form,
                'pos_tag': self.pos_tag,
                'feats': self.feats,
                'index': self.index,
                'binyan': self.binyan,
                'shoresh': self.shoresh}
    
    def setBinyan(self, binyan):
        self.binyan = binyan
        
    def setShoresh(self, shoresh):
        self.shoresh = shoresh