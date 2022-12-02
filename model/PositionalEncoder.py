import numpy as np
import torch

def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
    

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class SinusoidalEncoder(torch.nn.Module):
    """
    Sets up embedding layer for word sequences as well as for word positions.Both the layers are trainable.
    Returns embeddings of words which also contains the position(time) component
    """
    def __init__(self, emb_dim, max_len):
        """
        emb_dim : [int] embedding dimension for words
        max_len : [int] maxlen of input sentence (!!!start by 1)
        """  
        super(SinusoidalEncoder,self).__init__()
        self.emb_dim = emb_dim
        self.max_len = max_len
        n_position = max_len+1
        self.position_enc = torch.nn.Embedding(n_position, emb_dim, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(n_position, emb_dim)        

    def forward(self, p_sequences):
        '''input:
                p_sequences: (batch_size, n_node) 
            output:
                PE: (batch_size, n_node, dim_node)
        '''
        return self.position_enc(p_sequences.long())

if __name__=='__main__':

    device = torch.device('cuda')
    position_encoder = SinusoidalEncoder(emb_dim=3, max_len=3).to(device)

    position = torch.tensor([[0, 2],[2, 1]])

    print(position_encoder(position))