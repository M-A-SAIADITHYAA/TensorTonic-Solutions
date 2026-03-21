import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here

    K  = K.transpose(-2, -1)
    qkt = Q @K
    dk = Q.size(-1)
    qkt = qkt/math.sqrt(dk)
    res = torch.softmax(qkt,dim=-1)
    res1 = res @ V
    return res1
    
    pass