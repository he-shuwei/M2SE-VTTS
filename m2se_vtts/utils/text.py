import re

PAD = "<pad>"
EOS = "<EOS>"
UNK = "<UNK>"
SEG = "|"
RESERVED_TOKENS = [PAD, EOS, UNK]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)
EOS_ID = RESERVED_TOKENS.index(EOS)
UNK_ID = RESERVED_TOKENS.index(UNK)

RESERVED_TOKENS_BYTES = [bytes(PAD, "ascii"), bytes(EOS, "ascii")]

_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")
_ESCAPE_CHARS = set(u"\\_u;0123456789")

def strip_ids(ids, ids_to_strip):
    ids = list(ids)
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids

class TextEncoder(object):

    def __init__(self, num_reserved_ids=NUM_RESERVED_TOKENS):
        self._num_reserved_ids = num_reserved_ids

    @property
    def num_reserved_ids(self):
        return self._num_reserved_ids

    def encode(self, s):
        return [int(w) + self._num_reserved_ids for w in s.split()]

    def decode(self, ids, strip_extraneous=False):
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        return " ".join(self.decode_list(ids))

    def decode_list(self, ids):
        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < self._num_reserved_ids:
                decoded_ids.append(RESERVED_TOKENS[int(id_)])
            else:
                decoded_ids.append(id_ - self._num_reserved_ids)
        return [str(d) for d in decoded_ids]

    @property
    def vocab_size(self):
        raise NotImplementedError()

class ByteTextEncoder(TextEncoder):

    def encode(self, s):
        numres = self._num_reserved_ids
        return [c + numres for c in s.encode("utf-8")]

    def decode(self, ids, strip_extraneous=False):
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        numres = self._num_reserved_ids
        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < numres:
                decoded_ids.append(RESERVED_TOKENS_BYTES[int(id_)])
            else:
                decoded_ids.append(bytes([id_ - numres]))
        return b"".join(decoded_ids).decode("utf-8", "replace")

    def decode_list(self, ids):
        numres = self._num_reserved_ids
        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < numres:
                decoded_ids.append(RESERVED_TOKENS_BYTES[int(id_)])
            else:
                decoded_ids.append(bytes([id_ - numres]))
        return decoded_ids

    @property
    def vocab_size(self):
        return 2**8 + self._num_reserved_ids

class ByteTextEncoderWithEos(ByteTextEncoder):

    def encode(self, s):
        return super(ByteTextEncoderWithEos, self).encode(s) + [EOS_ID]

class TokenTextEncoder(TextEncoder):

    def __init__(self,
               vocab_filename,
               reverse=False,
               vocab_list=None,
               replace_oov=None,
               num_reserved_ids=NUM_RESERVED_TOKENS):
        super(TokenTextEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        if vocab_filename:
            self._init_vocab_from_file(vocab_filename)
        else:
            assert vocab_list is not None
            self._init_vocab_from_list(vocab_list)
        self.pad_index = self._token_to_id[PAD]
        self.eos_index = self._token_to_id[EOS]
        self.unk_index = self._token_to_id[UNK]
        self.seg_index = self._token_to_id[SEG] if SEG in self._token_to_id else self.eos_index

    def encode(self, s):
        sentence = s
        tokens = sentence.strip().split()
        if self._replace_oov is not None:
            tokens = [t if t in self._token_to_id else self._replace_oov
                        for t in tokens]
        ret = [self._token_to_id[tok] for tok in tokens]
        return ret[::-1] if self._reverse else ret

    def decode(self, ids, strip_eos=False, strip_padding=False):
        if strip_padding and self.pad() in list(ids):
            pad_pos = list(ids).index(self.pad())
            ids = ids[:pad_pos]
        if strip_eos and self.eos() in list(ids):
            eos_pos = list(ids).index(self.eos())
            ids = ids[:eos_pos]
        return " ".join(self.decode_list(ids))

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in seq]

    @property
    def vocab_size(self):
        return len(self._id_to_token)

    def __len__(self):
        return self.vocab_size

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, "ID_%d" % idx)

    def _init_vocab_from_file(self, filename):
        with open(filename) as f:
            tokens = [token.strip() for token in f.readlines()]

        def token_gen():
            for token in tokens:
                yield token

        self._init_vocab(token_gen(), add_reserved_tokens=False)

    def _init_vocab_from_list(self, vocab_list):
        def token_gen():
            for token in vocab_list:
                if token not in RESERVED_TOKENS:
                    yield token

        self._init_vocab(token_gen())

    def _init_vocab(self, token_generator, add_reserved_tokens=True):

        self._id_to_token = {}
        non_reserved_start_index = 0

        if add_reserved_tokens:
            self._id_to_token.update(enumerate(RESERVED_TOKENS))
            non_reserved_start_index = len(RESERVED_TOKENS)

        self._id_to_token.update(
            enumerate(token_generator, start=non_reserved_start_index))

        self._token_to_id = dict((v, k)
                                for k, v in self._id_to_token.items())

    def pad(self):
        return self.pad_index

    def eos(self):
        return self.eos_index

    def unk(self):
        return self.unk_index

    def seg(self):
        return self.seg_index

    def store_to_file(self, filename):
        with open(filename, "w") as f:
            for i in range(len(self._id_to_token)):
                f.write(self._id_to_token[i] + "\n")

    def sil_phonemes(self):
        return [p for p in self._id_to_token.values() if not p[0].isalpha()]
