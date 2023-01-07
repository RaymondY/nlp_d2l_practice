import collections
import math
import os
import random
import torch
from d2l import torch as d2l
from config import DefaultConfig

config = DefaultConfig()


# Implementation of the Vocab class is in d2l.py; for reference:
# class Vocab:
#     """Vocabulary for text."""
#     def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
#         """Defined in :numref:`sec_text-sequence`"""
#         # Flatten a 2D list if needed
#         if tokens and isinstance(tokens[0], list):
#             tokens = [token for line in tokens for token in line]
#         # Count token frequencies
#         counter = collections.Counter(tokens)
#         self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
#                                   reverse=True)
#         # The list of unique tokens
#         self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
#             token for token, freq in self.token_freqs if freq >= min_freq])))
#         self.token_to_idx = {token: idx
#                              for idx, token in enumerate(self.idx_to_token)}
#
#     def __len__(self):
#         return len(self.idx_to_token)
#
#     def __getitem__(self, tokens):
#         if not isinstance(tokens, (list, tuple)):
#             return self.token_to_idx.get(tokens, self.unk)
#         return [self.__getitem__(token) for token in tokens]
#
#     def to_tokens(self, indices):
#         if hasattr(indices, '__len__') and len(indices) > 1:
#             return [self.idx_to_token[int(index)] for index in indices]
#         return self.idx_to_token[indices]
#
#     @property
#     def unk(self):  # Index for the unknown token
#         return self.token_to_idx['<unk>']


# Penn Tree Bank (PTB) dataset
# The PTB dataset is a collection of texts from the Wall Street Journal.
# The dataset is divided into three subsets: training, validation, and test.
# Each subset contains a list of sentences separated by spaces, where each sentence is a list of words
# Treat each word as a token.
def read_ptb():
    data_path = config.data_path
    # with open(os.path.join(data_path, 'ptb.train.txt'), 'r') as f:
    #     lines = f.readlines()
    #     # print(f'# training sentences: {len(lines)}')
    #     # 42068
    #     output = [line.split('\n') for line in lines]
    # # return output

    with open(os.path.join(data_path, 'ptb.train.txt')) as f:
        raw_text = f.read()
        output = [line.split() for line in raw_text.split('\n')]

    # 42069; The last line is empty.
    return output


# build a vocabulary for the corpus
# any word that appears less than 10 times is replaced by the “<unk>” token.
def build_vocab(sentences, min_freq=10):
    vocab = d2l.Vocab(sentences, min_freq=min_freq)
    return vocab


# discard high-frequency words
# each word is discarded with a probability of 1 − sqrt(t/freq(w)), where t is a threshold hyperparameter
def subsample(sentences, vocab):
    # exclude unknown tokens ('<unk>')
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    # count the frequency for each word
    counter = collections.Counter([token for line in sentences for token in line])
    # total number of tokens
    num_tokens = sum(counter.values())

    def keep(token):
        # The subsampling probability for a given word is given by
        # P(w) = max(1 − sqrt(t/f(w)), 0), where t is a threshold hyperparameter and f(w) is the frequency of a word.
        # where t is a threshold value and f(w) is the frequency of a word w.
        return random.uniform(0, 1) < math.sqrt(config.freq_threshold / counter[token] * num_tokens)

    # Now we can discard all the high-frequency words
    return [[token for token in line if keep(token)] for line in sentences], counter


# compare the given word
def compare_counts(token, sentences, sub_sentences):
    print(f'# of "{token}": '
          f'before={sum([l.count(token) for l in sentences])}, '
          f'after={sum([l.count(token) for l in sub_sentences])}')


# Extracts all the center words and their context words from corpus
def get_centers_and_contexts(corpus, max_window_size):
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for center_i in range(len(line)):
            # Context window centered at 'center_i'
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(line), center_i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(center_i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


def get_similar_tokens(vocab, query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9)
    _, topk = torch.topk(cos, k=k + 1)
    topk = topk.cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')


def main():
    sentences = read_ptb()
    print(f'# sentences: {len(sentences)}')

    vocab = build_vocab(sentences)
    print(f'# tokens: {len(vocab)}')

    sub_sentences, counter = subsample(sentences, vocab)
    d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                                'count', sentences, sub_sentences)
    d2l.plt.show()

    # an example of high-frequency words
    compare_counts('the', sentences, sub_sentences)
    # an example of low-frequency words
    compare_counts('join', sentences, sub_sentences)

    # After subsampling, we map tokens to their indices for the corpus.
    corpus = [vocab[line] for line in sub_sentences]
    print(corpus[:3])


if __name__ == '__main__':
    main()
