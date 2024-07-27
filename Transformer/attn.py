from utils import reset_seed, rel_error
import torch
from torch import nn


def scaled_dot_product_no_loop_batch(
    query, key, value, mask = None
):
    """

    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. It uses
    Matrix-matrix multiplication to find the scaled weights and then matrix-matrix
    multiplication to find the final output.

    args:
        query: a Tensor of shape (N,K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension

        key:  a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension


        value: a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension


        mask: a Bool Tensor of shape (N, K, K) that is used to mask the weights
            used for computing weighted sum of values


    return:
        y: a tensor of shape (N, K, M) that contains the weighted sum of values

        weights_softmax: a tensor of shape (N, K, K) that contains the softmaxed
            weight matrix.

    """

    _, _, M = query.shape
    y = None
    weights_softmax = None
    ###############################################################################
    # TODO: This function performs same function as self_attention_two_loop_batch #
    # Implement this function using no loops.                                     #
    # For the mask part, you can ignore it for now and revisit it in the later part.
    # Given the shape of the mask is (N, K, K), and it is boolean with True values#
    # indicating  the weights that have to be masked and False values indicating  #
    # the weghts that dont need to be masked at that position. These masked-scaled#
    # weights can then be softmaxed to compute the final weighted sum of values   #
    # Hint: look at torch.bmm and torch.masked_fill                               #
    ###############################################################################
    # Replace "pass" statement with your code
    E = torch.bmm(query, key.transpose(1,2)) / (M**0.5)
    if mask is not None:
        ##########################################################################
        # TODO: Apply the mask to the weight matrix by assigning -1e9 to the     #
        # positions where the mask value is True, otherwise keep it as it is.    #
        ##########################################################################
        # Replace "pass" statement with your code
        E[mask] = -1e9
    # Replace "pass" statement with your code
    weights_softmax = nn.functional.softmax(E, dim=2)
    y = torch.bmm(weights_softmax, value)
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return y, weights_softmax


class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        super().__init__()

        """
        This class encapsulates the implementation of self-attention layer. We map 
        the input query, key, and value using MLP layers and then use 
        scaled_dot_product_no_loop_batch to the final output.
        
        args:
            dim_in: an int value for input sequence embedding dimension
            dim_q: an int value for output dimension of query and ley vector
            dim_v: an int value for output dimension for value vectors

        """
        self.q = None  # initialize for query
        self.k = None  # initialize for key
        self.v = None  # initialize for value
        self.weights_softmax = None
        ##########################################################################
        # TODO: This function initializes three functions to transform the 3 input
        # sequences to key, query and value vectors. More precisely, initialize  #
        # three nn.Linear layers that can transform the input with dimension     #
        # dim_in to query with dimension dim_q, key with dimension dim_q, and    #
        # values with dim_v. For each Linear layer, use the following strategy to#
        # initialize the weights:                                                #
        # If a Linear layer has input dimension D_in and output dimension D_out  #
        # then initialize the weights sampled from a uniform distribution bounded#
        # by [-c, c]                                                             #
        # where c = sqrt(6/(D_in + D_out))                                       #
        # Please use the same names for query, key and value transformations     #
        # as given above. self.q, self.k, and self.v respectively.               #
        ##########################################################################
        # Replace "pass" statement with your code
        # c = (6/(dim_in + dim_q))**0.5 
        self.q = torch.nn.Linear(dim_in, dim_q)
        
        # nn.init.uniform_(self.q.weight, -c, c)
        nn.init.xavier_uniform_(self.q.weight)
        
        self.k = torch.nn.Linear(dim_in, dim_q)
        # nn.init.uniform_(self.k.weight, -c, c)
        nn.init.xavier_uniform_(self.k.weight)
        
        # c = (6/(dim_in + dim_v))**0.5 
        self.v = torch.nn.Linear(dim_in, dim_v)
        # nn.init.uniform_(self.v.weight, -c, c)
        nn.init.xavier_uniform_(self.v.weight)
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(
        self, query, key, value, mask = None
    ):

        """
        An implementation of the forward pass of the self-attention layer.

        args:
            query of shape (N, K, M)
            key of shape (N, K, M)
            value of shape (N, K, M)
            mask of shape (N, K, K)
        return:
            y of shape (N, K, dim_v)
        """
        self.weights_softmax = (
            None  # weight matrix after applying self_attention_no_loop_batch
        )
        y = None
        ##########################################################################
        # TODO: Use the functions initialized in the init fucntion to find the   #
        # output tensors. Precisely, pass the inputs query, key and value to the #
        #  three functions iniitalized above. Then, pass these three transformed #
        # query,  key and value tensors to the self_attention_no_loop_batch to   #
        # get the final output. For now, dont worry about the mask and just      #
        # pass it as a variable in self_attention_no_loop_batch. Assign the value#
        # of output weight matrix from self_attention_no_loop_batch to the       #
        # variable self.weights_softmax                                          #
        ##########################################################################
        # Replace "pass" statement with your code
        q = self.q(query)
        
        k = self.k(key)
        v = self.v(value)

        y, self.weight_softmax = scaled_dot_product_no_loop_batch(q, k, v, mask)
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()

        """
        
        A naive implementation of the MultiheadAttention layer for Transformer model.
        We use multiple SelfAttention layers parallely on the same input and then concat
        them to into a single tensor. This Tensor is then passed through an MLP to 
        generate the final output. The input shape will look like (N, K, M) where  
        N is the batch size, K is the batch size and M is the sequence embedding  
        dimension.
        args:
            num_heads: int value specifying the number of heads
            dim_in: int value specifying the input dimension of the query, key
                and value. This will be the input dimension to each of the
                SingleHeadAttention blocks
            dim_out: int value specifying the output dimension of the complete 
                MultiHeadAttention block



        NOTE: Here, when we say dimension, we mean the dimesnion of the embeddings.
              In Transformers the input is a tensor of shape (N, K, M), here N is
              the batch size , K is the sequence length and M is the size of the
              input embeddings. As the sequence length(K) and number of batches(N)
              don't change usually, we mostly transform
              the dimension(M) dimension.
        """
        ##########################################################################
        # TODO: Initialize two things here:                                      #
        # 1.) Use nn.ModuleList to initialze a list of SingleHeadAttention layer #
        # modules.The length of this list should be equal to num_heads with each #
        # SingleHeadAttention layer having input dimension as dim_in, and query  #
        # , key, and value dimension as dim_out.                                 #
        # 2.) Use nn.Linear to map the output of nn.Modulelist block back to     #
        # dim_in. Initialize the weights using the strategy mentioned in         #
        # SelfAttention.                                                         #
        ##########################################################################
        # Replace "pass" statement with your code
        # c = (6/(dim_in + dim_out))**0.5 
        
        self.attn_heads = nn.ModuleList([
            SelfAttention(dim_in, dim_out, dim_out)
            for _ in range(num_heads)
        ])
        
        self.linear_backs = nn.Linear(num_heads*dim_out, dim_in)
        print(self.linear_backs.weight.T.shape)
        nn.init.xavier_uniform_(self.linear_backs.weight)
        
        # nn.init.uniform_(self.linear_backs.weight, -c, c)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(
        self, query, key, value, mask = None
    ):

        """
        An implementation of the forward pass of the MultiHeadAttention layer.

        args:
            query of shape (N, K, M) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

            key of shape (N, K, M) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

            value of shape (N, K, M) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

            mask of shape (N, K, K) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

        returns:
            y of shape (N, K, M)
        """
        ys = []
        ##########################################################################
        # TODO: You need to perform a forward pass through the MultiHeadAttention#
        # block using the variables defined in the initializing function. The    #
        # nn.ModuleList behaves as a list and you could use a for loop or list   #
        # comprehension to extract different elements of it. Each of the elements#
        # inside nn.ModuleList is a SingleHeadAttention that  will take the same #
        # query, key, value and mask tensors and you will get a list of tensors as
        # output. Concatenate this list if tensors and pass them through the     #
        # nn.Linear mapping function defined in the initialization step.         #
        ##########################################################################
        # Replace "pass" statement with your code
        for attn_head in self.attn_heads:
            sy = attn_head(query, key, value, mask)
            ys.append(sy)
        ys = torch.concat(ys, dim=-1)
        # print(ys.shape)
        # print(self.linear_backs.weight.T.shape)
        y = self.linear_backs(ys)
        # print(y.shape)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return y
    
reset_seed(0)
N = 2
num_heads = 2
K = 4
M = inp_emb_size = 4
out_emb_size = 8
# atten_multihead = MultiHeadAttention(num_heads, inp_emb_size, out_emb_size)
atten_multihead = torch.nn.MultiheadAttention(inp_emb_size, num_heads, batch_first=True)


for k, v in atten_multihead.named_parameters():
    # print(k, v.T.shape) # uncomment this to see the weight shape
    v.data.copy_(torch.linspace(-1.4, 1.3, steps=v.numel()).reshape(*v.shape))

query = torch.linspace(-0.4, 0.6, steps=N * K * M, requires_grad=True).reshape(
    N, K, M
)  # **to_double_cuda
key = torch.linspace(-0.8, 0.5, steps=N * K * M, requires_grad=True).reshape(
    N, K, M
)  # **to_double_cuda
value = torch.linspace(-0.3, 0.8, steps=N * K * M, requires_grad=True).reshape(
    N, K, M
)  # *to_double_cuda

query.retain_grad()
key.retain_grad()
value.retain_grad()

y_expected = torch.tensor(
    [
        [
            [-0.23104, 0.50132, 1.23367, 1.96603],
            [0.68324, 1.17869, 1.67413, 2.16958],
            [1.40236, 1.71147, 2.02058, 2.32969],
            [1.77330, 1.98629, 2.19928, 2.41227],
        ],
        [
            [6.74946, 5.67302, 4.59659, 3.52015],
            [6.82813, 5.73131, 4.63449, 3.53767],
            [6.86686, 5.76001, 4.65315, 3.54630],
            [6.88665, 5.77466, 4.66268, 3.55070],
        ],
    ]
)
dy_expected = torch.tensor(
    [[[ 0.56268,  0.55889,  0.55510,  0.55131],
         [ 0.43286,  0.42994,  0.42702,  0.42411],
         [ 2.29865,  2.28316,  2.26767,  2.25218],
         [ 0.49172,  0.48841,  0.48509,  0.48178]],

        [[ 0.25083,  0.24914,  0.24745,  0.24576],
         [ 0.14949,  0.14849,  0.14748,  0.14647],
         [-0.03105, -0.03084, -0.03063, -0.03043],
         [-0.02082, -0.02068, -0.02054, -0.02040]]]
)

y, _ = atten_multihead(query, key, value, need_weights=False)
print(y)
dy = torch.randn(*y.shape)  # , **to_double_cuda
print(y.shape)
print(dy_expected.shape)
y.backward(dy)
query_grad = query.grad
print("MultiHeadAttention error: ", rel_error(y_expected, y))
print("MultiHeadAttention error: ", rel_error(dy_expected, query_grad))



# muti_head_attn = MultiHeadAttention(2, 4, 8)
# muti_head_attn = MultiHeadAttention(2, 4, 4 // 2)
