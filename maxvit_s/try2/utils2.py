def get_block_input_sizes(input_size, n_layers):
    block_input_sizes = []
    input_size = get_conv_size(input_size)
    for _ in range(n_layers):
        input_size = get_conv_size(input_size)
        block_input_sizes.append(input_size)
    return block_input_sizes
    
def get_conv_size(input_size, kernel=3, stride=2, padding=1):
    return [
        (input_size[0] - kernel * 2*padding) // stride + 1,
        (input_size[1] - kernel * 2*padding) // stride + 1
    ]