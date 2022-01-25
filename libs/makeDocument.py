#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import bs4
from libs.corp_df import *


# In[2]:


def getSoup(fileDir):
    infile = open(fileDir, 'r')
    contents = infile.read()
    soup = BeautifulSoup(contents, 'xml')
    
    return soup


# In[3]:


def makeDocument(doc_id, fileDir):
    soup = getSoup(fileDir)
    
    paragraphs = []
    
    pgs = soup.find_all("paragraph")
    
    for pg in pgs:
        pg_attrs = pg.attrs
        pg_id = pg_attrs['id']
        sents = []
        for sent in pg.children:
            if type(sent) == bs4.element.Tag:
                sent_attrs = sent.attrs
                sent_id = sent_attrs['id']

                toks = []
                
                # append padding
                toks.append(Token(-1, '[CLS]'))
                
                for tok in sent.children:
                    if type(tok) == bs4.element.Tag:
                        tok_attrs = tok.attrs
                        tok_id = tok['id']
                        tok_surface = tok['surface']

                        toks.append(Token(tok_id, tok_surface))
                        
                # append padding
                toks.append(Token(-1, '[SEP]'))

                sents.append(Sentence(sent_id, toks))

        paragraphs.append(Paragraph(pg_id, sents))
        
        return Document(doc_id, paragraphs)


# In[ ]:




