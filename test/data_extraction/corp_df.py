class Document:
    def __init__(self, doc_id, paragraphs):
        self.doc_id = doc_id
        self.paragraphs = paragraphs
        
    def __iter__(self):
        return iter(self.paragraphs)
    
    def __str__(self):
        return '\n\n'.join([str(i) for i in self.paragraphs])
        
class Paragraph:
    def __init__(self, pg_id, sentences):
        self.pg_id = pg_id
        self.sentences = sentences
        
    def __iter__(self):
        return iter(self.sentences)
    
    def __str__(self):
        return '\n'.join([str(i) for i in self.sentences])
        
class Sentence:
    def __init__(self, sent_id, tokens):
        self.sent_id = sent_id
        self.tokens = tokens
        
    def __iter__(self):
        return iter(self.tokens)
    
    def __str__(self):
        return ' '.join([i.raw_tok for i in self.tokens])
        
class Token:
    def __init__(self, tok_id, raw_tok, lemmas=None, embedding=None):
        self.tok_id = tok_id
        self.raw_tok = raw_tok
        self.lemmas = lemmas if lemmas is not None else []
        self.embedding = embedding if embedding is not None else None
            
    def __iter__(self):
        return iter(self.lemmas)
    
    def __str__(self):
        return raw_tok
        
class Lemma:
    def __init__(self, lemma, form, pos_tag, feats, index):
        self.lemma = lemma
        self.form = form
        self.pos_tag = pos_tag
        self.feats = feats
        self.index = index
        self.binyan = None
        self.shoresh = None
        

