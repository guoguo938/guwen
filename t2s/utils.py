import io
import numpy as np
import torch
from collections import Counter
#OOV预设阈值
OOV_THRESHOLD = 5
def get_char2freq(path):
    char_counter = Counter()
    with open(path, encoding='utf-8') as f:
        for line in f:
            for ch in line.strip():
                char_counter[ch] += 1
    return dict(char_counter)


def load_vec(emb_path):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
    id2word = {v: k for k, v in word2id.items()}
    embeddings = torch.tensor(np.array(vectors))

    return embeddings, id2word, word2id

def accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean()

def get_nn(word_emb, tgt_emb, tgt_id2word, K=1):
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    for i, idx in enumerate(k_best):
      return tgt_id2word[idx]



def split_cn(sentence):
    result = []
    for w in sentence:
        result.append(w)
    return result