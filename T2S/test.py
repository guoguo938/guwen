import torch
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()

import numpy as np
from torch.utils.data import DataLoader
from .datasets import MyDataset
from .model import MyModel
from .utils import load_vec,get_char2freq,get_nn,split_cn
OOV_THRESHOLD=5
def test_output(direction="s2t", src_file=None,tgt_file=None,mapping_path=None,model_path=None,input_file=None,output_file=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if input_file is None or output_file is None:
        raise ValueError("必须提供 input_file 和 output_file 参数")
    if src_file is None or tgt_file is None or mapping_path is None:
        raise ValueError("必须提供向量文件参数")
    
    src_path=input_file
    src_emb,src_id2word,src_word2id=load_vec(src_file)
    tgt_emb,tgt_id2word,tgt_word2id=load_vec(tgt_file)
    word_emb=tgt_emb
    
   
    char2freq=get_char2freq(src_path)
    src_set = MyDataset(src_path, src_emb, src_word2id,src_id2word )
    
    src_loader = DataLoader(src_set, 1, False)
    bert_path=os.path.join(BASE_DIR,"model")
    model = MyModel(word_emb, mapping_path, bert_path).to(device)
    model = MyModel(word_emb, mapping_path, bert_path).to(device)
    model_path=os.path.join(BASE_DIR,model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds=[]
    total = 0
    for (src_emb, _, src_text) in src_loader:
        src_emb = src_emb.to(device)
        with torch.no_grad():
            o = model(x_s=src_emb).squeeze(0)
        # pred_ids = o.argmax(-1).cpu().numpy()
        # pred_text = "".join([id2word.get(i, "") for i in pred_ids])
        s=""
        for (i,ow),(j,char) in zip(enumerate(o),enumerate(src_text[0])):
            freq=char2freq.get(char,0)
            if freq<OOV_THRESHOLD:
                
                s+=char
            else:
                ow = ow.squeeze(0).squeeze(0).detach().cpu().numpy()
                w=get_nn(ow,np.eye(len(word_emb)),tgt_id2word)
                s+=w

        pred = split_cn(s)
        print(pred)
        all_preds.append(pred)
        
        total += 1
    # 保存结果
    output_file=os.path.join(output_file,"result.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for line in all_preds:
            f.write("".join(line) + "\n")

            

    print(f"✅ 转换完成，结果已保存到：{output_file}")
