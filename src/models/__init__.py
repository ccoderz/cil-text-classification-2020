"""Model definitions (one class per file) to define models."""
from .CharCNN import CharCNN
from .CharResCNN_GRU import CharResCNN_GRU
from .BERT_GRU import BERTGRU
from .CSR_Res_5_d_GRU import CSRRes5d
from .CSR_Res_5_GRU import CSRRes5

__all__ = ('CharCNN', 'CharResCNN_GRU', 'BERTGRU', 'CSR_Res_5_d_GRU', 'CSR_Res_5_GRU')


def get_model(name):
    if name == "CharCNN":
        return CharCNN
    elif name == "CharResCNN_GRU":
        return CharResCNN_GRU
    elif name == 'BERT_GRU':
        return BERT_GRU
    elif name == 'CSR_Res_5_d_GRU':
        return CSRRes5d
    elif name == 'CSR_Res_5_GRU':
        return CSRRes5
    else:
        raise NotImplementedError("No model named \"%s\"!" % name)