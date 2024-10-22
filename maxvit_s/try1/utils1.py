def get_block_input_sizes(input_size, n_layers):
    block_input_sizes = []
    current_size = get_conv_size(input_size, 3, 2, 1)
    for _ in range(n_layers):
        current_size = get_conv_size(current_size, 3, 2, 1)
        block_input_sizes.append(current_size)
    return block_input_sizes

def get_conv_size(input_size, kernel=3, stride=2, padding=1):
    return [
        (input_size[0] - kernel + 2 * padding) // stride + 1,
        (input_size[1] - kernel + 2 * padding) // stride + 1
    ]