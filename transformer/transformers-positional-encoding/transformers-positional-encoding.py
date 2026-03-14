import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    array = np.arange(seq_length).reshape(-1,1)
    pe = np.zeros((seq_length,d_model))

    for i in range(d_model):
        for j in range(seq_length):
            power1 = (2*(i//2))/d_model
            dem = 10000**power1
            res = 0
            if(i%2==0):
                res = np.sin(j/dem)
            else:
                res = np.cos(j/dem)
            pe[j][i] = res
            
    return pe   
    
    pass