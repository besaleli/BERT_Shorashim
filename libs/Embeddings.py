# imports ########################################
import torch
from transformers import AutoTokenizer, AutoModel
from libs.corp_df import *
##################################################

EMBEDDINGLOGDIR = 'logs/decode.txt'

def log_tokens(toks):
    with open(EMBEDDINGLOGDIR, 'a') as f:
        f.write(toks + '\n\n')
        f.close()
    
class Embedding:
    def __init__(self, modelName):
        self.tokenizer = AutoTokenizer.from_pretrained(modelName)
        self.model = AutoModel.from_pretrained(modelName, output_hidden_states=True)
        self.model.eval()
        
    def encode_sent(self, sent):
        tokens = [i.raw_tok for i in sent]
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        log_tokens(' '.join([str(i) for i in indexed_tokens]))
        log_tokens(self.tokenizer.decode(indexed_tokens))

        segments_ids = [1] * len(tokens)

        # convert tokens to tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        
        # retrieve hidden states
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)

            hidden_states = outputs[2]

        # dimensions of layers: [layer][token][unit]
        layers = torch.squeeze(torch.stack(hidden_states, dim=0), dim=1)

        # change dimensions to: [token][layer][unit]
        token_layers = layers.permute(1, 0, 2)

        token_embeddings = [token_layers[i] for i in range(len(tokens))]
        
        return tokens, indexed_tokens, token_embeddings
