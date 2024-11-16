import numpy as np


data = open("./input.txt").read().strip()

np.random.seed(42)

vocabs = set(data)
data_size = len(data)
vocab_size = len(vocabs)
sample_size = data_size

char_to_idx = {c:i for i,c in enumerate(vocabs)}
idx_to_char = {i:c for i,c in enumerate(vocabs)}

hidden_size = 96
seq_size = 5
lr = 1e-2

n,p = 0,0

input_weights_U = np.random.randn(hidden_size, vocab_size) * 0.02
hidden_weights_W = np.random.randn(hidden_size, hidden_size) * 0.02
hidden_bias = np.zeros((hidden_size, 1)) 

output_weight_V = np.random.randn(vocab_size, hidden_size) * 0.02
output_bias = np.zeros((vocab_size, 1))

def sample(hidden_state_prev, seed_ch, n):
    x = np.zeros((vocab_size, 1))
    x[seed_ch] = 1
    xs = []
    for _ in range(n):
        hidden_state_prev = np.tanh(input_weights_U @ x + hidden_weights_W @ hidden_state_prev + hidden_bias)
        y = output_weight_V @ hidden_state_prev + output_bias
        z = y - np.max(y)
        # softmax
        p = np.exp(z) / np.sum(np.exp(z))
        xi = np.argmax(p.ravel())
        x = np.zeros((vocab_size, 1))
        x[xi] = 1
        xs.append(xi)
    return xs

def step(inputs, targets, hidden_state_prev):
    xs, hidden_states, outputs, probabilities = {}, {}, {}, {}
    hidden_states[-1] = np.copy(hidden_state_prev)
    
    # forward pass
    loss = 0
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        input_char = inputs[t]
        target_char = targets[t]
        xs[t][input_char] = 1
        hidden_states[t] = np.tanh(input_weights_U @ xs[t] + hidden_weights_W @ hidden_states[t-1] + hidden_bias)
        outputs[t] = output_weight_V @ hidden_states[t] + output_bias
        z = outputs[t] - np.max(outputs[t])
        # softmax
        probabilities[t] = np.exp(z) / np.sum(np.exp(z))
        # CrossEntropyLoss softmax
        loss += -np.log(probabilities[t][target_char].item())
        
    input_weights_U_grad = np.zeros_like(input_weights_U)
    hidden_weights_W_grad = np.zeros_like(hidden_weights_W)
    hidden_bias_grad = np.zeros_like(hidden_bias)
    output_weight_V_grad = np.zeros_like(output_weight_V)
    output_bias_grad = np.zeros_like(output_bias)
    
    hidden_state_next_grad = np.zeros_like(hidden_states[0])
    
    for t in reversed(range(len(inputs))):
        output_grad = np.copy(probabilities[t])
        output_grad[targets[t]] -= 1
        output_weight_V_grad += output_grad @ hidden_states[t].T
        output_bias_grad += output_grad
        dh = output_weight_V.T @ output_grad + hidden_state_next_grad
        dhraw = (1 - hidden_states[t] * hidden_states[t]) * dh
        hidden_bias_grad += dhraw
        input_weights_U_grad += dhraw @ xs[t].T
        hidden_weights_W_grad += dhraw @ hidden_states[t-1].T
        hidden_state_next_grad = hidden_weights_W.T @ dhraw
        
    for param_grad in [input_weights_U_grad, hidden_weights_W_grad, output_weight_V_grad, output_bias_grad, hidden_bias_grad]:
        np.clip(param_grad, -5, 5, out=param_grad)
        
    return loss, input_weights_U_grad, hidden_weights_W_grad, hidden_bias_grad, output_weight_V_grad, output_bias_grad, hidden_states[len(inputs)-1]

smooth_loss = -np.log(1/vocab_size) * seq_size
while True:
    
    if n == 0 or p+seq_size+1>=len(data):
        hidden_state_prev = np.zeros((hidden_size, 1))
        p=0
    
    inputs = [char_to_idx[c] for c in data[p:p+seq_size]]
    targets = [char_to_idx[c] for c in data[p+1:p+seq_size+1]]
    
    if p==0 and n % 100 == 0:
        xs = sample(hidden_state_prev, inputs[0], sample_size)
        txt = ''.join(idx_to_char[xi] for xi in xs)
        print(f"===\n {txt} \n===")

    loss, input_weights_U_grad, hidden_weights_W_grad, hidden_bias_grad, output_weight_V_grad, output_bias_grad, hidden_state_prev = step(inputs, targets, hidden_state_prev)        
        
    for param, param_grad in zip([input_weights_U, hidden_weights_W, hidden_bias, output_weight_V, output_bias], [input_weights_U_grad, hidden_weights_W_grad, hidden_bias_grad, output_weight_V_grad, output_bias_grad]):
        param += -lr * param_grad

    p += seq_size
    n += 1
