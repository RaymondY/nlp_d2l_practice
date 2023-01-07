import torch
from torch.utils import data
from utils import *
from config import DefaultConfig

config = DefaultConfig()


class PTBDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        # check validity
        assert len(centers) == len(contexts) == len(negatives)

        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return self.centers[index], self.contexts[index], self.negatives[index]

    def __len__(self):
        return len(self.centers)


# Sample noise words according to a predefined distribution.
class RandomGenerator:
    def __init__(self, sampling_weights):
        self.sampling_weights = sampling_weights
        self.population = list(range(1, len(sampling_weights) + 1))
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache k random sampling results
            self.candidates = random.choices(
                self.population, weights=self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


# Return noise words in negative sampling
def get_negatives(all_contexts, vocab, counter, K):
    # Sampling weights for words with indices 0, 1, ...
    # (index 0 is the excluded unknown token) in the vocabulary
    # According to the "word2vec" paper,
    # the sampling probability P (w) of a noise word w is set to
    # its relative frequency in the dictionary raised to the power of 0.75 (Mikolov et al., 2013).
    sampling_weights = [counter[vocab.to_tokens(i)] ** 0.75 for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            # The random sampling is performed with the sampling weights
            neg = generator.draw()
            # Exclude the context words from the negative samples
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return torch.tensor(centers).reshape(-1, 1), torch.tensor(contexts_negatives), \
        torch.tensor(masks), torch.tensor(labels)


def load_data_ptb(batch_size, max_window_size, num_noise_words):
    sentences = read_ptb()
    vocab = build_vocab(sentences)
    sub_sentences, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in sub_sentences]

    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, vocab, counter, num_noise_words)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_loader = data.DataLoader(dataset, batch_size, shuffle=True,
                                  collate_fn=batchify, num_workers=0)

    return data_loader, vocab


def main():
    x_1 = (1, [2, 2], [3, 3, 3, 3])
    x_2 = (1, [2, 2, 2], [3, 3])
    batch = batchify((x_1, x_2))
    names = ['centers', 'contexts_negatives', 'masks', 'labels']
    for name, data in zip(names, batch):
        print(name, '=', data)

    data_loader, vocab = load_data_ptb(config.batch_size, config.max_window_size,
                                       config.num_noise_words)
    for batch in data_loader:
        for name, data in zip(['centers', 'contexts_negatives', 'masks', 'labels'], batch):
            print(name, 'shape:', data.shape)
        break


if __name__ == '__main__':
    main()
