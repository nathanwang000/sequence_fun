import time, math, torch

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def pad_timeseries_tensor(x, max_length): # pad with 0
    input_size = x.shape[-1]
    tensor = torch.zeros(max_length, 1, input_size)
    tensor[:len(x)] = x
    return tensor
                
