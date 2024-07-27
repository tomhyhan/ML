import json

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


if __name__ =="__main__":
    data = get_toy_data()

    num_examples = 4
    for q, a in zip(
        data["inp_expression"][:num_examples],
        data["out_expression"][:num_examples]
        ):
        print("Expression: " + q + " Output: " + a)
    token_dict = generate_token_dict()
    seq = prepocess_input_sequence("BOS POSITIVE 03 add POSITIVE 06 EOS", token_dict)
    print(seq)