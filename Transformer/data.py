import json
import torch
from sklearn.model_selection import train_test_split
from position_encoder import position_encoding_sinusoid

# 1. get data
def get_toy_data(path: str = "final_data.json"):
    return json.load(open(path))


# 2. generate token dict
def generate_token_dict():
    SPECIAL_TOKENS = ["POSITIVE", "NEGATIVE", "add", "subtract", "BOS", "EOS"]
    vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] + SPECIAL_TOKENS
    token_dict = {v: i for i, v in enumerate(vocab)}
    return token_dict    

# 3. preprocess_input_sequence
def prepocess_input_sequence(input_str, token_dict) :
    spc_tokens = ["POSITIVE", "NEGATIVE", "add", "subtract", "BOS", "EOS"]
    out = []
    for token in input_str.split():
        if token in spc_tokens:
            out.append(token_dict[token])
        else:
            for digit in token:
                out.append(token_dict[digit])
    return out

class AddSubDataset(torch.utils.data.Dataset):
    def __init__(self, input_seqs, target_seqs, convert_str_to_token, emp_dim, pos_enc):
        
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.convert_str_to_token = convert_str_to_token
        self.emp_dim = emp_dim
        self.pos_enc = pos_enc
            
    def preprocess(self, inp):
        return prepocess_input_sequence(inp, self.convert_str_to_token)
    
    def __getitem__(self, idx):
        inp = self.input_seqs[idx]
        target = self.target_seqs[idx]
        
        inp_tokens = torch.tensor(self.preprocess(inp))
        target_tokens = torch.tensor(self.preprocess(target))
        
        inp_seq_len = len(inp_tokens)
        inp_tokens_pos = self.pos_enc(inp_seq_len, self.emp_dim)
    
        target_seq_len = len(target_tokens)
        target_tokens_pos = self.pos_enc(target_seq_len, self.emp_dim)
        
        return inp_tokens, inp_tokens_pos[0], target_tokens, target_tokens_pos[0]
        
    def __len__(self):
        return len(self.input_seqs)
    
if __name__ =="__main__":
    data = get_toy_data()
    
    inp_expression = data["inp_expression"]
    out_expression = data["out_expression"]

    X_train, X_test, y_train, y_test = train_test_split(inp_expression, out_expression, test_size=0.1, random_state=0)
    
    emp_dim = 4
    tokens = generate_token_dict()
    train_set = AddSubDataset(X_train, y_train, tokens, emp_dim, position_encoding_sinusoid)
    
    print(train_set[0])
    print(len(train_set))
    
    
    # num_examples = 4
    # for q, a in zip(
    #     data["inp_expression"][:num_examples],
    #     data["out_expression"][:num_examples]
    #     ):
    #     print("Expression: " + q + " Output: " + a)
    # token_dict = generate_token_dict()
    # seq = prepocess_input_sequence("BOS POSITIVE 03 add POSITIVE 06 EOS", token_dict)
    # print(seq)
    # print(len(data["inp_expression"]))
    # print(len(data["out_expression"]))