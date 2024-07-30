import os
from sklearn.model_selection import train_test_split
import torch
import torch.utils

from position_encoder import position_encoding_sinusoid
from data import get_toy_data, generate_token_dict, AddSubDataset
from transformers import Transformers
from train import train_model, val, inference

device = "cpu"

num_heads = 8
emp_dim = 64
feedforward_dim = 256
num_enc_layer = 6
num_dec_layer = 6
dropout = 0.2
vocab_len = 16

num_epochs = 70
batch_size = 256
lr = 6e-4
loss_function = torch.nn.functional.cross_entropy

data = get_toy_data()

inp_expression = data["inp_expression"]
out_expression = data["out_expression"]

X_train, X_test, y_train, y_test = train_test_split(inp_expression, out_expression, test_size=0.1, random_state=0)

tokens = generate_token_dict()
train_set = AddSubDataset(X_train, y_train, tokens, emp_dim, position_encoding_sinusoid)

validation_set = AddSubDataset(X_test, y_test, tokens, emp_dim, position_encoding_sinusoid)


train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=False, drop_last=True
)

val_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=batch_size, shuffle=False, drop_last=True
)

model = Transformers(num_heads, emp_dim, feedforward_dim, dropout, num_enc_layer, num_dec_layer, vocab_len, device)

try:
    weights_path = os.path.join(os.getcwd(), "transformer.pt")

    model.load_state_dict(torch.load(weights_path))
    
    num_epochs = 2
except:
    print("weights not found")

model = train_model(model, train_loader, val_loader, loss_function, lr, num_epochs, batch_size, device)


weights_path = os.path.join(os.getcwd() , "transformer.pt")
torch.save(model.state_dict(), weights_path)

# print(len(next(iter(train_loader))))
# print(next(iter(val_loader)).shape)

accuracy = val(model, val_loader, loss_function, device)
print(accuracy)

# Now let's test it

item = next(iter(val_loader))

que, que_pos, ans, ans_pos = item
reverse_tokens = {v:k for k, v in tokens.items()}

model = model.to(device)
que = que.to(device)
que_pos = que_pos.to(device)
ans = ans.to(device)
ans_pos = ans_pos.to(device)

for i in range(5):
    que_exp = que[i:i+1, :] 
    que_exp_pos = que_pos[i:i+1, :] 
    ans_exp_pos = ans_pos[i:i+1, :]

    expression = [reverse_tokens[w.item()] for w in que_exp[0]] 

    print("Input: ", ' '.join(expression))

    ans_seq_len = 5

    exp_out, _ = inference(model, que_exp, que_exp_pos, ans_exp_pos, ans_seq_len, device)

    result = []
    for i in range(1, ans_seq_len):
        w = reverse_tokens[exp_out[0, i].item()] 
        if w == "EOS":
            break
        result.append(w)
    print("Output: ", " ".join(result))
