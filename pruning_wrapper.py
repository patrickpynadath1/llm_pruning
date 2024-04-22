import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
import tqdm 


# sets the smallest (in terms of magnitude) weights to 0
def get_sparse_weights(data, percentile=.25): 
    flat = data.flatten()
    values, _ = flat.abs().sort(descending=False)
    idx_to_get = int(percentile * flat.size(0))
    cutoff_val = values[idx_to_get]
    mask = torch.where(data.abs() > cutoff_val, 1, 0)
    return data * mask


class DeepSub(nn.Module):
    def __init__(self, input_dim, inner_rank, output_dim) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_features=input_dim, out_features=inner_rank)
        self.act = nn.GELU()
        self.l2 = nn.Linear(in_features=inner_rank, out_features=output_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x
    

def truncated_svd(W, l):
    """Compress the weight matrix W of an inner product (fully connected) layer
    using truncated SVD.
    Parameters:
    W: N x M weights matrix
    l: number of singular values to retain
    Returns:
    Ul, L: matrices such that W \approx Ul*L
    """

    U, s, V = torch.linalg.svd(W)

    Ul = U[:, :l]
    sl = s[:l]
    V = V.t()
    Vl = V[:l, :]

    SV = torch.mm(torch.diag(sl), Vl)
    return Ul, SV


def get_svd_ffn(w1, w2, l, bias=False): 
    ul1, sv1 = truncated_svd(w1, l)
    ul2, sv2 = truncated_svd(w2, l)

    w1_ffn_sv = nn.Linear(sv1.size(1), sv1.size(0), bias=bias)
    w1_ffn_sv.weight.data = sv1
    w1_ffn_ul = nn.Linear(ul1.size(1), ul.size(0), bias=bias)
    w1_ffn_ul.weight.data = ul1

    w2_ffn_sv = nn.Linear(sv2.size(1), sv2.size(0), bias=bias)
    w2_ffn_sv.weight.data = sv2
    w2_ffn_ul = nn.Linear(ul2.size(1), ul2.size(0), bias=bias)
    w2_ffn_ul.weight.data = ul2
    svd_module = nn.Sequential(w1_ffn_sv, w1_ffn_ul, w2_ffn_sv, w2_ffn_ul)
    return svd_module
    
def train_deep_sub(deep_sub,
                   gt_module,
                   training_iter, 
                  input_size):
    criterion = MSELoss()
    optimizer = Adam(deep_sub.parameters(), lr=0.001)
    for _ in range(training_iter):
        rand_batch = torch.randn((512, input_size))
        optimizer.zero_grad()
        output = deep_sub(rand_batch)
        # true val calc 
        x = gt_module[0](rand_batch)
        true_val = gt_module[1](x, rand_batch)
        loss = criterion(output, true_val)
        loss.backward()
        optimizer.step()
    return deep_sub

    
def train_deep_sub_svd(deep_sub,
                       svd_module,
                        gt_module,
                        training_iter, 
                        input_size, 
                        batch_size=512,
                        lr=.001):
    criterion = MSELoss()
    optimizer = Adam(deep_sub.parameters(), lr=lr)
    for _ in tqdm.tqdm(range(training_iter)):
        rand_batch = torch.randn((batch_size, input_size))
        optimizer.zero_grad()
        output = deep_sub(rand_batch)
        svd_output = svd_module(rand_batch)

        # true val calc 
        x = gt_module[0](rand_batch)
        true_val = gt_module[1](x, rand_batch)

        loss = criterion(output+svd_output, true_val)
        loss.backward()
        optimizer.step()
    return deep_sub


# function for pruning attention heads 
# core idea: for all the attention heads, keep track of the attention matrix score across 
# some set of inputs. 
# at the end, compute the fro norm of all the resulting matrices. For each layer, prune all but the top 
# k attention heads
def prune_attention_heads(bert_model, 
                          test_df, 
                          top_k = 3): 
    return bert_model

# main idea: replace the ffn in the bert model with a deeper and narrow network
# we train this network to approximate the original ffn using random noise as input
# strategies: 
# 1. vanilla_ffn: replace the ffn with a deeper and narrow network
# 2. svd_ffn: replace the ffn with a svd compressed network -- no training of network 
# 3. svd_ffn_train: replace the ffn with a svd compressed network and train the network
# 4. sparse_ffn: replace the ffn with a sparse network -- no training of network 
def replace_bert_ffn(bert_model, 
                     strategy='vanilla_ffn', 
                     train_iter = 1000, 
                     save_dir='substitute_networks',
                     inner_rank=200,
                     percentile=.25, 
                     svd_bias=True): 
    for layer_counter, cur_layer in enumerate(bert_model.bert.encoder.layer): 
        intermediate_weights = cur_layer.intermediate.dense.weight.data 
        output_weights = cur_layer.output.dense.weight.data

        sparse_intermediate = get_sparse_weights(intermediate_weights, percentile)
        sparse_output = get_sparse_weights(output_weights, percentile)

        cur_layer.intermediate.dense.weight.data = sparse_intermediate
        cur_layer.output.dense.weight.data = sparse_output
        
        gt_module = nn.Sequential(
                                    cur_layer.intermediate, 
                                    cur_layer.output)

        if strategy == 'vanilla_ffn': 
            deep_sub_intermediate = DeepSub(input_dim=intermediate_weights.size(0), 
                                            inner_rank=inner_rank, 
                                            output_dim=inner_rank)
            deep_sub_output = DeepSub(input_dim=inner_rank, 
                                      inner_rank=inner_rank, 
                                      output_dim=intermediate_weights.size(0))
            deep_sub_module = nn.Sequential(deep_sub_intermediate, deep_sub_output)
            deep_sub_module = train_deep_sub(deep_sub_module, 
                                             gt_module, 
                                             train_iter, 
                                             intermediate_weights.size(0))
            # replacing the big ffn with the deep sub module 
            cur_layer.intermediate.dense = deep_sub_module[0]
            cur_layer.output.dense = deep_sub_module[1]

        elif strategy == 'svd_ffn': 
            svd_module = get_svd_ffn(sparse_intermediate, 
                                     sparse_output, 
                                     inner_rank)
            # TODO: how do you access modules in a sequential module? 
            cur_layer.intermediate.dense = nn.sequential(svd_module[0], svd_module[1])
            cur_layer.output.dense = nn.sequential(svd_module[2], svd_module[3])

        elif strategy == 'svd_ffn_train': 
            svd_module = get_svd_ffn(intermediate_weights, 
                                     output_weights, 
                                     inner_rank, bias=svd_bias)

            svd_module = train_deep_sub(svd_module, 
                                            gt_module, 
                                            train_iter, 
                                            intermediate_weights.size(0))

            cur_layer.intermediate.dense = nn.sequential(svd_module[0], svd_module[1])
            cur_layer.output.dense = nn.sequential(svd_module[2], svd_module[3])
        
        elif strategy == 'sparse_ffn': 
            pass # do nothing; already replaced the data with sparsified matrix
    bert_model.save_pretrained(f"{save_dir}_{strategy}")
    print("Saved Model")
    return bert_model