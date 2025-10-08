import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from .datasets import MyDataset
from .model import MyModel
from .utils import load_vec,get_char2freq,get_nn,split_cn
OOV_THRESHOLD=5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()

def run_evaluation(direction="s2t", src_e=None,tgt_e=None,mapping_path=None,model_path=None,src_path=None,tgt_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # if direction == "s2t":
    #     cn_emb, cn_id2word, cn_word2id = load_vec('./data/S2T/vectors-cn.txt')
    #     tw_emb, tw_id2word, tw_word2id = load_vec('./data/S2T/vectors-tw.txt')
    #     mapping_path = "./data/S2T/best_mapping.pth"
    #     src_path, tgt_path = "./data/testset.s.txt", "./data/testset.t.txt"
    #     src_emb=cn_emb
    #     word_emb = tw_emb
    #     id2word = tw_id2word
    # else:
    #     cn_emb, cn_id2word, cn_word2id = load_vec('./data/T2S/vectors-cn.txt')
    #     tw_emb, tw_id2word, tw_word2id = load_vec('./data/T2S/vectors-tw.txt')
    #     mapping_path = "./data/T2S/best_mapping.pth"
    #     src_path, tgt_path = "./data/testset.t.txt", "./data/testset.s.txt"
    #     src_emb=tw_emb
    #     word_emb = cn_emb
    #     id2word = cn_id2word
    # char2freq=get_char2freq(src_path)
    # src_set = MyDataset(src_path, src_emb, cn_word2id if direction=="s2t" else tw_word2id,
    #                     cn_id2word if direction=="s2t" else tw_id2word)
    # tgt_set = MyDataset(tgt_path, word_emb, tw_word2id if direction=="s2t" else cn_word2id,
    #                     tw_id2word if direction=="s2t" else cn_id2word)
    
    
    src_emb,src_id2word,src_word2id= load_vec (src_e)
    tgt_emb,tgt_id2word,tgt_word2id= load_vec (tgt_e)
    word_emb=tgt_emb

    char2freq=get_char2freq(src_path)
    src_set = MyDataset(src_path, src_emb, src_word2id,src_id2word )
    tgt_set = MyDataset(tgt_path, tgt_emb, tgt_word2id,tgt_id2word )
    src_loader = DataLoader(src_set, 1, False)
    tgt_loader = DataLoader(tgt_set, 1, False)

    
    
    
    
    
    
    
    
    src_loader = DataLoader(src_set, 1, False)
    tgt_loader = DataLoader(tgt_set, 1, False)
    bert_path=os.path.join(BASE_DIR,"model")
    model = MyModel(word_emb, mapping_path, bert_path).to(device)
    model = MyModel(word_emb, mapping_path, bert_path).to(device)
    model_path=os.path.join(BASE_DIR,model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    accs=0
    accall = 0
    total = 0
    preds = []
    refes = []
    for (src_emb, _, src_text), (_, src_ids, tgt_text) in zip(src_loader, tgt_loader):
        src_emb = src_emb.to(device)
        with torch.no_grad():
            o = model(x_s=src_emb).squeeze(0)
        #pred_ids = o.argmax(-1).cpu().numpy()
        s=""
        for (i,ow),(j,char) in zip(enumerate(o),enumerate(src_text[0])):
            freq=char2freq.get(char,0)
            if freq<OOV_THRESHOLD:
                print("OOV跳过：",char)
                s+=char
            else:
                ow = ow.squeeze(0).squeeze(0).detach().cpu().numpy()
                w=get_nn(ow,np.eye(len(word_emb)),tgt_id2word)
                s+=w

        pred = split_cn(s)
        refe = split_cn(str(tgt_text[0]))
        preds.append(pred)
        refes.append(refe)
        cor=0
        for i,j in zip(refe,pred):
            if i==j:
                cor+=1
        acc=cor/len(tgt_text[0])
        if acc<0.8:
            print(acc)
            print("原文：",src_text[0])
            print("模型预测：",pred)
            print("参考：",refe)
        accs+=acc
        
        total += 1

    print(f"Accuracy word: {accs / total:.4f}")
