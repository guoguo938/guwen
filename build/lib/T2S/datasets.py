import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, path, word_emb, word2id, id2word):
        super(MyDataset, self).__init__()
        self.lines = open(path).readlines()

        self.word_emb = word_emb
        self.word2id = word2id
        self.id2word = id2word

    def __getitem__(self, index):
        MAX_LEN = 512  # 与模型配置一致
        line = self.lines[index].rstrip('\n')[:MAX_LEN]  # 显式截断+去除换行符
        #line = self.lines[index]   

        for i, w in enumerate(line):
            #w检测是否在词表中，不在0填充
            if w not in self.word2id:
                x = torch.zeros(768).unsqueeze(0)
                xo = torch.zeros(1)
            else:
                x = self.word_emb[self.word2id[w]].unsqueeze(0)
                xo = torch.tensor([self.word2id[w]])

            if i == 0:
                x_s = x
                xo_s = xo
            else:
                x_s = torch.cat([x_s, x], dim=0)
                xo_s = torch.cat([xo_s, xo], dim=0)

        return x_s, xo_s, line

    def __len__(self):
        return len(self.lines)