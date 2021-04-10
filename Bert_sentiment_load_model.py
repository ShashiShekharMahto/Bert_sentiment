import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

bert = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

# init_token = tokenizer.cls_token
# eos_token = tokenizer.sep_token
# pad_token = tokenizer.pad_token
# unk_token = tokenizer.unk_token
# init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
# eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
# pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
# unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)
init_token_idx = 101
eos_token_idx = 102
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]

        _, hidden = self.rnn(embedded)

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        output = self.out(hidden)

        return output

class Model:
    def __init__(self):
        pass
    def predict(self,sentence):
        sentiment_model.eval()
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:max_input_length - 2]
        indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)
        prediction = torch.sigmoid(sentiment_model(tensor))
        return prediction.item()

sentiment_model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)
sentiment_model.load_state_dict(torch.load('model/tut6-model.pt',map_location=torch.device('cpu')))

m = Model()
#print(sentiment.predict("okay"))
#print(m.predict("okay"))
def get_model():
    return m