from sklearn.model_selection import train_test_split
import torch
import torch.utils

from position_encoder import position_encoding_sinusoid
from data import get_toy_data, generate_token_dict, AddSubDataset
from transformers import Transformers
from train import train_model

num_heads = 4
emp_dim = 16
feedforward_dim = 64
num_enc_layer = 4
num_dec_layer = 4
dropout = 0.1
vocab_len = 16

num_epochs = 3
batch_size = 128
lr = 1e-4
loss_function = torch.nn.CrossEntropyLoss

data = get_toy_data()

inp_expression = data["inp_expression"]
out_expression = data["out_expression"]

X_train, X_test, y_train, y_test = train_test_split(inp_expression, out_expression, test_size=0.1, random_state=0)

tokens = generate_token_dict()
train_set = AddSubDataset(X_train, y_train, tokens, emp_dim, position_encoding_sinusoid)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=False, drop_last=True
)

model = Transformers(num_heads, emp_dim, feedforward_dim, dropout, num_enc_layer, num_dec_layer, vocab_len)

train_model(train_loader, model, lr, loss_function, )