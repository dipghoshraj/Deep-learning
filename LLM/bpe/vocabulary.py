

class Vocabulary:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def build(self, tokens):
        tokens = sorted(tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def token2id(self, token):
        return self.token_to_id.get(token, None)

    def id2token(self, idx):
        return self.id_to_token.get(idx, None)
