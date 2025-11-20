import torch

def create_sliding_window(x, window_size, dim=0):
    if dim != 0:
        x = x.transpose(0, dim)

    x_out = []
    for i in range(0, x.size(0)-window_size+1):
        x_out.append(x[i: i+window_size])
    x_out = torch.stack(x_out, dim=0)

    if dim != 0:
        x_out = x_out.transpose(dim+1, 1)
    return x_out