import heapq
from collections import defaultdict, Counter
import json, os

class FastBPETokenizer:
    def __init__(self, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]):
        self.vocab = None
        self.merges = []
        self.merge_ranks = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = special_tokens




    def get_vocab_from_corpus(self, corpus):
        """
        Build initial vocab from raw corpus: words -> freq
        Represent words as tuples of chars instead of strings
        """
        vocab = Counter()
        for line in corpus:
            for word in line.strip().split():
                symbols = tuple(word) + ("</w>",)  # tuple of chars
                vocab[symbols] += 1
        return vocab

    def get_pair_stats(self, vocab):
        """
        Compute frequency of symbol pairs in vocab
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """
        Merge a given pair in vocab efficiently
        """
        new_vocab = {}
        bigram = pair
        replacement = (pair[0] + pair[1],)  # tuple with merged symbol

        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(pair[0] + pair[1])  # merge
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def train(self, corpus, vocab_size=32000):
        """
        Train BPE with a priority queue for pair merges
        """
        print("Building initial vocab...")
        vocab = self.get_vocab_from_corpus(corpus)
        print(f"Initial vocab words: {len(vocab)}")

        # Compute initial pair stats
        pair_stats = self.get_pair_stats(vocab)
        heap = [(-freq, pair) for pair, freq in pair_stats.items()]
        heapq.heapify(heap)

        merges = []
        tokens = set(ch for word in vocab for ch in word)

        max_merges = vocab_size - len(tokens)
        print(f"Target vocab size={vocab_size}, initial={len(tokens)}, max merges={max_merges}")

        for i in range(max_merges):
            # Pop the most frequent pair
            if not heap:
                break
            freq, pair = heapq.heappop(heap)
            freq = -freq

            # If frequency is stale (doesn't match vocab anymore), skip
            if pair not in pair_stats or pair_stats[pair] != freq:
                continue

            # Merge vocab
            vocab = self.merge_vocab(pair, vocab)
            merges.append(pair)

            # Update pair stats incrementally
            pair_stats = self.get_pair_stats(vocab)
            for p, f in pair_stats.items():
                heapq.heappush(heap, (-f, p))

            if i % 100 == 0:
                print(f"Merge {i}: {pair} ({freq})")

        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.vocab = tokens.union(set(self.special_tokens)).union(set(a + b for a, b in merges))

        self.token_to_id = {token: idx for idx, token in enumerate(sorted(self.vocab))}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}


        print(f"Training done. Final vocab size={len(self.vocab)}")


    def tokenize_word(self, word):
        """
        Apply merges to a single word
        """
        symbols = list(word) + ["</w>"]
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            candidate_pairs = [p for p in pairs if p in self.merge_ranks]
            if not candidate_pairs:
                break
            best = min(candidate_pairs, key=lambda p: self.merge_ranks[p])
            i = pairs.index(best)
            symbols = symbols[:i] + [best[0] + best[1]] + symbols[i + 2 :]
        return symbols

    def tokenize(self, text):
        return [tok for word in text.split() for tok in self.tokenize_word(word)]
    
    def tokenize_to_ids(self, text):
        tokens = self.tokenize(text)
        ids = []
        ids.append(self.token_to_id.get("<bos>", None))  # Ensure <bos> is in vocab
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id.get("<unk>", 0))
        ids.append(self.token_to_id.get("<eos>", 1))  # Add <eos> at the end
        return ids
    
    def decode_from_ids(self, ids):
        tokens = [self.id_to_token[i] for i in ids]
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        vocab_path = os.path.join(path, "vocab.json")
        merges_path = os.path.join(path, "merges.json")

        with open(vocab_path, "w") as f:
            json.dump(self.token_to_id, f)
        with open(merges_path, "w") as f:
            json.dump(self.merges, f)

    def load(self, path):
        with open(f"{path}/vocab.json", "r") as f:
            self.token_to_id = json.load(f)
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            self.vocab = set(self.token_to_id.keys())
        with open(f"{path}/merges.json", "r") as f:
            self.merges = [tuple(pair) for pair in json.load(f)]
            self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
