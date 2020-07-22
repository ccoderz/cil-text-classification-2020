"""Model definitions (one class per file) to define models."""
from .CharCNN import CharCNN
from .CharResCNN_GRU import CharResCNN_GRU
from .BERT_GRU import BERTGRU

__all__ = ('CharCNN', 'CharResCNN_GRU', 'BERTGRU', )


def get_model(name):
    if name == "CharCNN":
        return CharCNN
    elif name == "CharResCNN_GRU":
        return CharResCNN_GRU
    elif name == 'BERT_GRU':
        return BERT_GRU
    else:
        raise NotImplementedError("No model named \"%s\"!" % name)