import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .datasets import MyDataset
from .model import MyModel
from .utils import load_vec
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()

def train_model(direction="s2t", epochs=1, batch_size=1,src_e=None,tgt_e=None,mapping_path=None,dataset=None,output=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # if direction == "s2t":
    #     s_emb, _, _ = load_vec('./data/S2T/vectors-cn.txt')
    #     t_emb, t_id2word, t_word2id = load_vec('./data/S2T/vectors-tw.txt')
    #     mapping_path = "./data/S2T/best_mapping.pth"
    #     train_path, val_path = "./data/trainset.t.txt", "./data/testset.s.txt"
    #     word_emb = t_emb
    #     id2word, word2id = t_id2word, t_word2id
    # else:   #t2s
    #     s_emb, s_id2word, s_word2id = load_vec('./data/T2S/vectors-cn.txt')
    #     t_emb, _, _ = load_vec('./data/T2S/vectors-tw.txt')
    #     mapping_path = "./data/T2S/best_mapping.pth"
    #     train_path, val_path = "./data/trainset.s.txt", "./data/testset.t.txt"
    #     word_emb = s_emb
    #     id2word, word2id = s_id2word, _word2id
    
    s_emb,s_id2word,s_word2id=load_vec(src_e)
    t_emb,t_id2word,t_word2id=load_vec(tgt_e)
    word_emb=t_emb

    train_set = MyDataset(dataset, t_emb, t_word2id, t_id2word)

    train_loader = DataLoader(train_set, batch_size, True, drop_last=True)
    bert_path=os.path.join(BASE_DIR,"model")
    model = MyModel(word_emb, mapping_path, bert_path).to(device)
    
    # Freeze W, train decoder
    for name, param in model.named_parameters():
        param.requires_grad = True

    for param in model.W.parameters():
        param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    writer = SummaryWriter()
    it=0
    for epoch in range(epochs):
        model.train()
        id=0
        total_loss = 0
        for y_s, yo_s, _ in train_loader:
            y_s, yo_s = y_s.to(device), yo_s.to(device)
            pred = model(y_s=y_s).squeeze(0)
            loss = criterion(pred, yo_s.squeeze(0).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/tra', loss.item(), it)
            total_loss += loss.item()
            it+=1
            id+=1
            if id % 1000 == 0:
                print("train loss:", loss.item())

        print(f"Epoch [{epoch+1}/{epochs}] train_loss={total_loss/len(train_loader):.4f}")
        save_path = os.path.join(output, f'param-{direction}.ckpt')

        torch.save(model.state_dict(),save_path)
    print("Training complete.")
    return save_path
    