import transformers as ppb
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from tqdm.notebook import tqdm
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import roc_auc_score

def cos_sim(a, b):
    ret = dot(a, b) / (norm(a) * norm(b))
    if np.isnan(ret) == True:
        return 0.0
    return ret

model_path = 'DeepPavlov/rubert-base-cased'

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, model_path)

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

import numpy as np
def as_matrix(sequences, max_len=None):
    tokenized = [tokenizer.encode(x, max_length=512, add_special_tokens=True) for x in sequences]
    max_len = 0
    for i in tokenized:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized])
    
    return padded

def get_features(model, batch, device):
    attention_mask = np.where(batch != 0, 1, 0)
    input_ids = torch.tensor(batch, dtype = torch.long).to(device)  
    attention_mask = torch.tensor(attention_mask, dtype = torch.long).to(device)
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    return last_hidden_states[0][:,0,:].detach().cpu().numpy()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

df = pd.read_csv('troll_data_cleaned.csv').dropna()

question = np.array([])
answer = np.array([])
batch_size = 64
for start in tqdm(range(0, len(df), batch_size)):
    question = np.append(question, get_features(model, as_matrix(df['question'].tolist()[start : start + batch_size]), device))
    answer = np.append(answer, get_features(model, as_matrix(df['answer'].tolist()[start : start + batch_size]), device))
    
question = np.reshape(question, (len(df), -1))
answer = np.reshape(answer, (len(df), -1))
similarity = np.array([])

for i in range(len(question)):
    similarity = np.append(similarity, cos_sim(question[i], answer[i]))

print(roc_auc_score(df['trollolo'], 1.0 - similarity))