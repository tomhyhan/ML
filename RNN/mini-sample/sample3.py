import numpy as np

np.random.seed(42)

data = open("./input.txt").read().strip()
vocabs = list(set(data))
n_data = len(data)
n_vocabs = len(vocabs)
n_sample = n_data

hidden_size = 96
seq_len = 5
lr = 1e-2

ch_to_idx = {c:i for i, c in enumerate(vocabs)}
idx_to_ch = {i:c for i, c in enumerate(vocabs)}

input_weights = np.random.randn(hidden_size, n_vocabs) * 0.02

hidden_weights = np.random.randn(hidden_size, hidden_size) * 0.02
hidden_bias = np.zeros((hidden_size, 1))

output_weights = np.random.randn(n_vocabs, hidden_size) * 0.02
output_bias = np.zeros((n_vocabs, 1))

def sample(h, input_seed, n_sample):
    x = np.zeros((n_vocabs, 1))    
    x[input_seed] = 1
    xs = []
    
    for _ in range(n_sample):
        h = np.tanh(input_weights @ x + hidden_weights @ h + hidden_bias)
        y = output_weights @ h + output_bias
        z = y - np.max(y)
        p = np.exp(z) / np.sum(np.exp(z))
        xi = np.argmax(p.ravel())
        x = np.zeros((n_vocabs, 1))    
        x[xi] = 1
        xs.append(xi)
    
    return xs

def step(inputs, targets, hidden_state_prev):
    xs, hidden_states, outputs, ps = {}, {}, {}, {}
    hidden_states[-1] = np.copy(hidden_state_prev)
    
    loss = 0
    for t in range(len(inputs)):
        input_i = ch_to_idx[inputs[t]]
        target_i = ch_to_idx[targets[t]]
        x = np.zeros((n_vocabs, 1))
        x[input_i] = 1
        xs[t] = x
        hidden_states[t] = np.tanh(input_weights @ xs[t] + hidden_weights @ hidden_states[t-1] + hidden_bias)
        y = output_weights @ hidden_states[t] + output_bias
        outputs[t] = y
        z = y - np.max(y)
        p = np.exp(z) / np.sum(np.exp(z))
        ps[t] = p
        loss += -np.log(p[target_i].item())
        
    input_weights_grad = np.zeros_like(input_weights)
    hidden_weights_grad = np.zeros_like(hidden_weights)
    hidden_bias_grad = np.zeros_like(hidden_bias)
    output_weights_grad = np.zeros_like(output_weights)
    output_bias_grad = np.zeros_like(output_bias)
    hidden_state_next_grad = np.zeros_like(hidden_states[-1])
    
    for t in reversed(range(len(inputs))):
        target_i = ch_to_idx[targets[t]]
        output_grad = np.copy(ps[t])
        output_grad[target_i] -= 1
        output_bias_grad += output_grad
        output_weights_grad += output_grad @ hidden_states[t].T 
        dh =  output_weights.T @ output_grad + hidden_state_next_grad
        dhTanh = (1 - hidden_states[t] * hidden_states[t]) * dh
        input_weights_grad += dhTanh @ xs[t].T 
        hidden_weights_grad += dhTanh @ hidden_states[t-1].T
        hidden_bias_grad += dhTanh
        hidden_state_next_grad = hidden_weights.T @ dhTanh 

    for param_grad in [input_weights_grad, hidden_weights_grad, hidden_bias_grad, output_weights_grad, output_bias_grad]:
        np.clip(param_grad, -5, 5, param_grad)

    return loss, input_weights_grad, hidden_weights_grad, hidden_bias_grad, output_weights_grad, output_bias_grad, hidden_states[-1]

p, n = 0, 0
smooth_loss = -np.log(1/n_vocabs) * seq_len
while True:
    if p+seq_len+1 >= n_data or n == 0:
        hidden_state_prev = np.zeros((hidden_size, 1))
        p = 0
    
    inputs = data[p:p+seq_len]
    outputs = data[p+1:p+seq_len+1]

    if n % 100 == 0 and p == 0:
        xs = sample(hidden_state_prev, ch_to_idx[inputs[0]], n_sample)
        txt = ''.join(idx_to_ch[i] for i in xs)
        print(f"===\n P{txt} \n===")
        
    loss, input_weights_grad, hidden_weights_grad, hidden_bias_grad, output_weights_grad, output_bias_grad, hidden_state_prev = step(inputs, outputs, hidden_state_prev)
    
    for param, param_grad in zip([input_weights, hidden_weights, hidden_bias, output_weights, output_bias], [input_weights_grad, hidden_weights_grad, hidden_bias_grad, output_weights_grad, output_bias_grad]):
        param += -lr * param_grad

    n += 1
    p += seq_len
    