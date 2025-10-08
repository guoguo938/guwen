import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()


class MyModel(nn.Module):
    def __init__(self, word_emb,mapping_path,bert_path):
        super(MyModel, self).__init__()
        
        self.W = nn.Linear(768, 768, bias=False)
        W_path = torch.from_numpy(torch.load(mapping_path,weights_only=False))
        self.W.weight = nn.Parameter(W_path)
        ###12层
        self.M1 = nn.Linear(768, 768, bias=False)
        self.M2 = nn.Linear(768, 768, bias=False)
        self.M3 = nn.Linear(768, 768, bias=False)
        self.M4 = nn.Linear(768, 768, bias=False)
        self.M5 = nn.Linear(768, 768, bias=False)
        self.M6 = nn.Linear(768, 768, bias=False)
        self.M7 = nn.Linear(768, 768, bias=False)
        self.M8 = nn.Linear(768, 768, bias=False)
        self.M9 = nn.Linear(768, 768, bias=False)
        self.M10 = nn.Linear(768, 768, bias=False)
        self.M11 = nn.Linear(768, 768, bias=False)
        self.M = nn.Linear(768, 768, bias=False)

        self.H = nn.Linear(len(word_emb),768,bias=False)
        self.H.weight = nn.Parameter(word_emb)

        
        self.Bert = RobertaModel.from_pretrained(bert_path, local_files_only=True)
        


    def forward(self, x_s=None, y_s=None):
        if x_s is None:
            x_hat_s = self.W(y_s.to(torch.float32))
        else:
            x_hat_s = x_s.to(torch.float32)

        
        outputs = self.Bert(inputs_embeds = x_hat_s, output_hidden_states=True)
        hidden_states = outputs[2][1:] 
        yo_hat_s=None
        
        h1, h2, h3, h4, h5, h6 = hidden_states[0], hidden_states[1], hidden_states[2], hidden_states[3], hidden_states[4], hidden_states[5]
        h7, h8, h9, h10, h11, h12 = hidden_states[6], hidden_states[7], hidden_states[8], hidden_states[9], hidden_states[10], hidden_states[11]

        y1, y2, y3, y4, y5, y6 = self.M1(h1), self.M2(h2), self.M3(h3), self.M4(h4), self.M5(h5), self.M6(h6)
        y7, y8, y9, y10, y11, y12 = self.M7(h7), self.M8(h8), self.M9(h9), self.M10(h10), self.M11(h11), self.M(h12)

        y = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12]
        if x_s is not None:
            yo_hat_s = self.H(y[5].to(torch.float64))

            # voting
            for i in range(4,6):
                yo_hat_s = self.H(y[i].to(torch.float64))

                sm = F.softmax(yo_hat_s, dim=2)
                p = sm.squeeze(0).max(dim=1).values
                
                T=0.996
                if p.mean() > T:
                    break
        else:
            
            yo_hat_s = self.H(y[5].to(torch.float64))

            
        return yo_hat_s
