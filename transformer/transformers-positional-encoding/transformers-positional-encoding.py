import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    position = np.arange(seq_length).reshape(seq_length,1)
    div = 1/(10000**(np.arange(0,d_model,2)/d_model))

    pe = np.zeros((seq_length,d_model))

    pe[:,0::2] = np.sin(position*div)
    pe[:,1::2] = np.cos(position*div)

    return pe  
    
    pass