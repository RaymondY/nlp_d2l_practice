import torch
from d2l import torch as d2l


class DefaultConfig:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)

    d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip', '319d85e578af0cdc590547f26231e4e31cdf1e42')
    data_path = d2l.download_extract('ptb')

    freq_threshold = 1e-4
    max_window_size = 5
    num_noise_words = 5

    batch_size = 512
    embed_size = 100
    num_epochs = 5
    lr = 2e-3


