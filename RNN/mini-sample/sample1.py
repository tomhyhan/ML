import numpy as np
np.random.seed(42)

def main():
    
    data = open("./input.txt", 'r').read().strip()
    chars = list(set(data))
    data_size = len(data)
    vocab_size = len(chars)
    n_sample_chars = data_size
    print(f"data has {data_size} total characters and {vocab_size} unique characters")
    char_to_idx = {c:i for i, c in enumerate(chars)}
    idx_to_char = {i:c for i, c in enumerate(chars)}

    hidden_size = 96
    seq_length = 5
    learning_rate = 1e-2

    # model parameter weights
    input_weights_U = np.random.randn(hidden_size, vocab_size) * 0.02
    hidden_weights_W = np.random.rand(hidden_size, hidden_size) * 0.02
    hidden_bias = np.zeros((hidden_size, 1))

    output_weight_V = np.random.rand(vocab_size, hidden_size) * 0.02
    output_bias = np.zeros((vocab_size, 1))

    def step(inputs, targets, hidden_state_prev):
        
        xs, hidden_states, outputs, probabilities = {}, {}, {}, {}
        hidden_states[-1] = np.copy(hidden_state_prev)
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
            # CrossEntropyLoss
            loss += -np.log(probabilities[t][target_char].item())
            
        # back propagation
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
            
        return loss, input_weights_U_grad, hidden_weights_W_grad, output_weight_V_grad, hidden_bias_grad, output_bias_grad, hidden_states[len(inputs) - 1]

    def sample(h, seed_i, n):
        x = np.zeros((vocab_size, 1))
        x[seed_i] = 1
        xs = []
        for _ in range(n):
            h = np.tanh(input_weights_U @ x + hidden_weights_W @ h + hidden_bias)
            y = output_weight_V @ h + output_bias
            z = y - np.max(y)
            p = np.exp(z) / np.sum(np.exp(z))
            # random sampling
            # ix = np.random.choice(vocab_size, p=p.ravel())
            # argmax version
            ix = np.argmax(p.ravel())
            x = np.zeros((vocab_size, 1))
            x[ix] = 1
            xs.append(ix)
        return xs
    
    n, p = 0, 0
    smooth_loss = -np.log(1.0/vocab_size)*seq_length
    while True:
        if p+seq_length + 1 >= len(data) or n ==0:
            hidden_state_prev = np.zeros((hidden_size, 1))
            p=0
            
        inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]
        
        if n % 100 == 0 and p == 0:
            sample_is = sample(hidden_state_prev, inputs[0], n_sample_chars)
            # print(sample_is)
            txt = "".join(idx_to_char[i] for i in sample_is)
            print(f"=====\n {txt} \n=====")
        
        loss, input_weights_U_grad, hidden_weights_W_grad, output_weight_V_grad, hidden_bias_grad, output_bias_grad, hidden_state_prev = step(inputs, targets, hidden_state_prev)
        
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 100 == 0 and p == 0:
            print(f'iter {n}, loss: {smooth_loss}')
        for param, param_grad in zip([input_weights_U, hidden_weights_W, output_weight_V, hidden_bias, output_bias],[input_weights_U_grad, hidden_weights_W_grad, output_weight_V_grad, hidden_bias_grad, output_bias_grad]):
            param += -learning_rate * param_grad
        
        n += 1
        p += seq_length
        

    

if __name__ == "__main__":
    main()