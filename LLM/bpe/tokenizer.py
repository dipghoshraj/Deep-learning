

import re
import collections

from LLM.bpe.byte_encoder import ByteEncoder
from LLM.bpe.vocabulary import Vocabulary

class Tokenizer:
    def __init__(self, byte_encoder: ByteEncoder):
        self.byte_encoder = byte_encoder
        self.vocab = None  # Vocabulary object
        self.merges = []   # List of merges (tuple pairs)
        self.merge_ranks = {}


    def get_vocab_from_corpus(self, corpus):
        vocab = collections.Counter()
        for line in corpus:
            words = re.findall(r'\S+', line)
            for word in words:
                tokens = list(word) + ['</w>']
                token_str = ' '.join(tokens)
                vocab[token_str] += 1
        return vocab
    
    def get_stats(self, vocab):
        pairs = collections.Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        new_vocab = {}
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        return new_vocab
    

    def train(self, corpus, num_merges=10000):
        print("Encoding corpus to byte-level unicode...")
        corpus = [self.byte_encoder.encode(text) for text in corpus]
        print("Building initial vocab...")
        vocab = self.get_vocab_from_corpus(corpus)
        print(f"Initial vocab size: {len(vocab)}")
        merges = []
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            merges.append(best)
            if i % 1000 == 0:
                print(f"Merge {i}: {best}")
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        # Build vocab tokens
        tokens = set()
        for word in vocab:
            tokens.update(word.split())
        self.vocab = Vocabulary()
        self.vocab.build(tokens)
        print(f"Training done. Final vocab size: {len(tokens)}")


    def tokenize_word(self, word):
        tokens = list(word)
        while True:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            candidate_pairs = [pair for pair in pairs if pair in self.merge_ranks]
            if not candidate_pairs:
                break
            best_pair = min(candidate_pairs, key=lambda p: self.merge_ranks[p])
            i = pairs.index(best_pair)
            tokens = tokens[:i] + [''.join(best_pair)] + tokens[i+2:]
        return tokens
    

    def tokenize(self, text):
        encoded = self.byte_encoder.encode(text)
        words = re.findall(r'\S+', encoded)
        all_tokens = []
        for word in words:
            tokens = self.tokenize_word(word)
            all_tokens.extend([self.vocab.token2id(t) for t in tokens if self.vocab.token2id(t) is not None])
        return all_tokens

    def detokenize(self, token_ids):
        tokens = [self.vocab.id2token(tid) for tid in token_ids]
        text = ''.join(tokens).replace('</w>', '')
        return self.byte_encoder.decode(text)