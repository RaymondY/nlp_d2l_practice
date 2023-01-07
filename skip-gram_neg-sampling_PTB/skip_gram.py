import math
import torch
from torch import nn
from tqdm import tqdm
from d2l import torch as d2l
from utils import *
from neg_sample import *
from config import DefaultConfig

config = DefaultConfig()


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed_v = nn.Embedding(vocab_size, embed_size)
        self.embed_u = nn.Embedding(vocab_size, embed_size)
        # Initialize the weights of the embedding layer
        nn.init.xavier_uniform_(self.embed_v.weight)
        nn.init.xavier_uniform_(self.embed_u.weight)

    def forward(self, center, contexts_and_negatives):
        v = self.embed_v(center)
        u = self.embed_u(contexts_and_negatives)
        # torch.bmm: Batch Matrix-Matrix Product of matrices stored in input and mat2.
        # pred = torch.bmm(v.unsqueeze(1), u.permute(0, 2, 1)).squeeze()
        pred = torch.bmm(v, u.permute(0, 2, 1))
        return pred


class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, mask=None):
        return nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, weight=mask, reduction='none').mean(dim=1)


def train(net, data_loader, num_epochs, lr, device):
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss',
    #                         xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    loss = SigmoidBCELoss()
    for epoch in range(num_epochs):
        # num_batches = len(data_loader)
        with tqdm(data_loader) as tepoch:
            for i, batch in enumerate(tepoch):

                center, context_negative, mask, label = [
                    data.to(device) for data in batch]

                pred = net(center, context_negative)
                l = loss(pred.reshape(label.shape).float(), label.float(), mask) / mask.sum(dim=1) * mask.shape[1]

                optimizer.zero_grad()
                l.sum().backward()
                optimizer.step()

                metric.add(l.sum(), l.numel())
                # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                #     animator.add(epoch + (i + 1) / num_batches,
                #                  (metric[0] / metric[1],))

                tepoch.set_postfix(epoch=epoch + 1, loss=metric[0] / metric[1])


def main():
    data_loader, vocab = load_data_ptb(config.batch_size, config.max_window_size,
                                       config.num_noise_words)
    net = SkipGramModel(len(vocab), config.embed_size)
    train(net, data_loader, config.num_epochs, config.lr, config.device)

    get_similar_tokens(vocab, 'chip', 10, net.embed_v)


if __name__ == '__main__':
    main()
    # sentences = read_ptb()
    # print(f'# sentences: {len(sentences)}')
    #
    # vocab = build_vocab(sentences)
    # print(f'# tokens: {len(vocab)}')
    #
    # sub_sentences, counter = subsample(sentences, vocab)
    #
    # # After subsampling, we map tokens to their indices for the corpus.
    # corpus = [vocab[line] for line in sub_sentences]
    # tiny_dataset = [list(range(7)), list(range(7, 10))]
    # print('dataset', tiny_dataset)
    # for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    #     print('center', center, 'has contexts', context)
    # all_centers, all_contexts = get_centers_and_contexts(corpus, 5)


