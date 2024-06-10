def compute_numeric_gradients(fx, x, dout=None, h=1e-7):
    """
        Compute numeric gradients using center difference
    """
    flat_x = x.continuous
    pass