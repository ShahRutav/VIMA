from torch.nn import Embedding as _Embedding


class Embedding(_Embedding):
    @property
    def output_dim(self):
        return self.embedding_dim
