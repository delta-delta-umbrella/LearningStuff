"""
This script is still pretty rough, need to work on it. Will upload as it is being completed.

T_T

"""

from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer)

from torch import nn
import torch
from scipy.spatial.distance import cosine

class MaskedLM(nn.Module):
    def __init__(self):
        super(MaskedLM, self).__init__()

        self.bert_layer = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)

        self.dropout = nn.Dropout(0.1)

    def forward(self, seq, attn_masks):

        loss, logits, output = self.bert_layer(seq, attention_mask=attn_masks, masked_lm_labels=seq)

        return loss, logits, output

tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

# model = MaskedLM()

# checkpoint = torch.load('D:/Dan/PythonProjects/SciBERT_CORD19/Models/MLM_model.pth')
# model.load_state_dict(checkpoint['model_state_dict'])

# Prediction
inference_file = torch.load('D:/Dan/PythonProjects/SciBERT_CORD19/model.pth')
embs = MaskedLM()
embs.load_state_dict(inference_file['model_state_dict'])

text = "Hydroxychloroquine is effective against the coronavirus."
marked_text = "[CLS] " + text + " [SEP]"
tokenized_text = tokenizer.tokenize(marked_text)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

for tup in zip(tokenized_text, indexed_tokens):
    print("{:<12} {:>6,}".format(tup[0], tup[1]))

tokens_tensor = torch.tensor([indexed_tokens])
attn_ids = [1] * len(tokenized_text)
attn_mask = torch.tensor([attn_ids])

embs.eval()

with torch.no_grad():
    loss, logits, output = embs(tokens_tensor, attn_mask)

print(output[0].unsqueeze(-1).size())

embedded_layers = output[0].unsqueeze(-1)

#for i in range(1, 12):
#   embedded_layers = torch.cat((embedded_layers, output[i].unsqueeze(-1)), dim=3)
#print(embedded_layers.size())

token_embeddings = torch.stack(output[0:12], dim=0)
print(token_embeddings.size())

token_embeddings = torch.squeeze(token_embeddings, dim=1)

print(token_embeddings.size())

# reorganize to [# tokens, # layers, # features]
token_embeddings = token_embeddings.permute(1, 0, 2)

print(token_embeddings.size())

# Use last 4 layers, therefore, 4x768 = 3072
# Stores the token vectors, with shape [13 x 3,072]

token_vecs_cat = []
# For each token in the sentence...
for token in token_embeddings:
    # `token` is a [12 x 768] tensor

    # Concatenate the vectors (that is, append them together) from the last
    # four layers.
    # Each layer vector is 768 values, so `cat_vec` is length 3,072.
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

    # Use `cat_vec` to represent `token`.
    token_vecs_cat.append(cat_vec)

print('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))



