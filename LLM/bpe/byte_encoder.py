
class ByteEncoder:
    def __init__(self):
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def _bytes_to_unicode(self):

        bs = list(range(ord('!'), ord('~') + 1)) + \
             list(range(ord('¡'), ord('¬') + 1)) + \
             list(range(ord('®'), ord('ÿ') + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    
    def encode(self, text: str) -> str:
        text_bytes = text.encode('utf-8')
        return ''.join(self.byte_encoder[b] for b in text_bytes)
    
    def decode(self, text: str) -> str:
        text_bytes = [self.byte_decoder[c] for c in text]
        return bytes(text_bytes).decode('utf-8', errors='replace')

