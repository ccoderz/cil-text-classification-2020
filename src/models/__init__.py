"""Model definitions (one class per file) to define models."""
from .CharCNN import CharCNN
from .CharResCNN_GRU import CharResCNN_GRU
from .BERT_GRU import BERTGRU
from .CSR_Res_4_d_GRU import CSRRes4d
from .CSR_Res_4_GRU import CSRRes4

__all__ = ('CharCNN', 'CharResCNN_GRU', 'BERTGRU', 'CSR_Res_4_d_GRU', 'CSR_Res_4_GRU')


def get_model(name):
    if name == "CharCNN":
        return CharCNN
    elif name == "CharResCNN_GRU":
        return CharResCNN_GRU
    elif name == 'BERT_GRU':
        return BERT_GRU
    elif name == 'CSR_Res_4_d_GRU':
        return CSRRes4d
    elif name == 'CSR_Res_4_GRU':
        return CSRRes4
    else:
        raise NotImplementedError("No model named \"%s\"!" % name)