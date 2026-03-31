import pickle
from copy import deepcopy

import numpy as np

class IndexedDataset:
    def __init__(self, path, num_cache=1):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = np.load(f"{path}.idx", allow_pickle=True).item()['offsets']
        self.cache = []
        self.num_cache = num_cache
        self._pid = None

    def _open_file(self):
        import os
        if self.data_file is not None:
            self.data_file.close()
        self.data_file = open(f"{self.path}.data", 'rb', buffering=-1)
        self._pid = os.getpid()

    def _ensure_file(self):
        import os
        if self.data_file is None or self._pid != os.getpid():
            self._open_file()

    def check_index(self, i):
        if i < 0 or i >= len(self.data_offsets) - 1:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['data_file'] = None
        state['_pid'] = None
        state['cache'] = []
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getitem__(self, i):
        self.check_index(i)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        self._ensure_file()
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        item = pickle.loads(b)
        if self.num_cache > 0:
            self.cache = ([(i, deepcopy(item))] + self.cache)[:self.num_cache]
        return item

    def __len__(self):
        return len(self.data_offsets) - 1

class IndexedDatasetBuilder:
    def __init__(self, path):
        self.path = path
        self.out_file = open(f"{path}.data", 'wb')
        self.byte_offsets = [0]

    def add_item(self, item):
        s = pickle.dumps(item)
        bytes = self.out_file.write(s)
        self.byte_offsets.append(self.byte_offsets[-1] + bytes)

    def finalize(self):
        self.out_file.close()
        np.save(open(f"{self.path}.idx", 'wb'), {'offsets': self.byte_offsets})
