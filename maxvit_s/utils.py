def make_block_input_shapes(input_size, n_blocks):
    shapes = []
    block_input_shape = conv_out_size(input_size, 3, 2, 1)
    for _ in range(n_blocks):
        block_input_shape = conv_out_size(block_input_shape, 3, 2, 1)
        shapes.append(block_input_shape)
    return shapes

def conv_out_size(input_size, kernel, stride, padding):
    return (
        (input_size[0] - kernel + 2*padding) // stride + 1,
        (input_size[1] - kernel + 2*padding) // stride + 1,
    )