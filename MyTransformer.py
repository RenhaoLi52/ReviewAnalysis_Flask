# 文件名: model.py

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0 if n_layers < 2 else dropout
        )
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        hidden = self.dropout(hidden)
        output = self.out(hidden)
        return output

# 创建模型实例并加载模型状态字典
def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    model = BERTGRUSentiment(
        bert=bert,
        hidden_dim=256,
        output_dim=5,
        n_layers=2,
        bidirectional=True,
        dropout=0.25
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('./models/model_bert.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model, tokenizer, device

# 定义预测函数
def predict_stars(texts, model, tokenizer, device, max_len=512):
    predictions = []
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = model(input_ids)
            _, predicted = torch.max(outputs, dim=1)
            predictions.append(predicted.cpu().item() + 1)  # 加1以匹配原始评分（1-5）
    return predictions
