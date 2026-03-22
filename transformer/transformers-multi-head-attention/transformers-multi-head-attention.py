import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:

    B, L, d_model = Q.shape
    assert d_model % num_heads == 0

    d_k = d_model // num_heads

    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    def split_heads(x):
        return x.reshape(B, L, num_heads, d_k).transpose(0, 2, 1, 3)

    Q = split_heads(Q)
    K = split_heads(K)
    V = split_heads(V)

    scores = Q @ K.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(d_k)

    weights = softmax(scores, axis=-1)
    out = weights @ V

    out = out.transpose(0, 2, 1, 3)
    out = out.reshape(B, L, d_model)

    out = out @ W_o

    return out