import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
import tqdm
import numpy as np
import json
reg_hook = torch.nn.modules.module.register_module_forward_hook



# sets the smallest (in terms of magnitude) weights to 0
def get_sparse_weights(data, percentile=0.25):
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


class DeepSubOutput(DeepSub):
    def __init__(self, input_dim, inner_rank, output_dim) -> None:
        super().__init__(input_dim, inner_rank, output_dim)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x, orig_input):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        x = self.ln(x + orig_input)
        return x


def truncated_svd(W, q):
    """Compress the weight matrix W of an inner product (fully connected) layer
    using truncated SVD.
    Parameters:
    W: N x M weights matrix
    l: number of singular values to retain
    Returns:
    Ul, L: matrices such that W \approx Ul*L
    """

    U, s, V = torch.svd_lowrank(W, q)

    Ul = U[:, :q]
    sl = s[:q]
    V = V.t()
    Vl = V[:q, :]

    SV = torch.mm(torch.diag(sl), Vl)
    return Ul, SV


def get_svd_ffn(w1, w2, l, bias=False, scaling_info=None, layer_idx=None, scaling_type=None):
    if scaling_info != None:
        if scaling_type == "grad": 
            raw_grad_mean_intermediate =  scaling_info["intermediate_first_moment"][layer_idx].sum(dim=-1)
            scaling_intermediate = torch.diag(raw_grad_mean_intermediate) 
            raw_grad_mean_output = scaling_info["output_first_moment"][layer_idx].sum(dim=-1)
            scaling_output = torch.diag(raw_grad_mean_output)
            w1 = torch.matmul(scaling_intermediate, w1)
            w2 = torch.matmul(scaling_output, w2)
            
            ul1, sv1 = truncated_svd(w1, l)
            ul2, sv2 = truncated_svd(w2, l)

            ul1 = torch.matmul(torch.inverse(scaling_intermediate), ul1)
            ul2 = torch.matmul(torch.inverse(scaling_output), ul2)
        else: 
            raw_grad_mean_intermediate =  scaling_info["intermediate_first_moment"][layer_idx].sum(dim=0)
            scaling_intermediate = torch.diag(raw_grad_mean_intermediate) 
            raw_grad_mean_output = scaling_info["output_first_moment"][layer_idx].sum(dim=0)
            scaling_output = torch.diag(raw_grad_mean_output)
            w1 = torch.matmul(w1, scaling_intermediate)
            w2 = torch.matmul(w2, scaling_output)
            
            ul1, sv1 = truncated_svd(w1, l)
            ul2, sv2 = truncated_svd(w2, l)

            sv1 = torch.matmul(sv1, torch.inverse(scaling_intermediate))
            sv2 = torch.matmul(sv2, torch.inverse(scaling_output)) 
    else: 
        ul1, sv1 = truncated_svd(w1, l)
        ul2, sv2 = truncated_svd(w2, l)

    w1_ffn_sv = nn.Linear(sv1.size(1), sv1.size(0), bias=bias)
    w1_ffn_sv.weight.data = sv1.contiguous()
    w1_ffn_ul = nn.Linear(ul1.size(1), ul1.size(0), bias=bias)
    w1_ffn_ul.weight.data = ul1.contiguous()

    w2_ffn_sv = nn.Linear(sv2.size(1), sv2.size(0), bias=bias)
    w2_ffn_sv.weight.data = sv2.contiguous()
    w2_ffn_ul = nn.Linear(ul2.size(1), ul2.size(0), bias=bias)
    w2_ffn_ul.weight.data = ul2.contiguous()
    svd_module = nn.Sequential(w1_ffn_sv, w1_ffn_ul, w2_ffn_sv, w2_ffn_ul)
    return svd_module


def train_deep_sub(deep_sub, gt_module, training_iter, input_size):
    criterion = MSELoss()
    optimizer = Adam(deep_sub.parameters(), lr=0.001)
    pg_bar = tqdm.tqdm(range(training_iter))
    for _ in pg_bar:
        rand_batch = torch.randn((512, input_size))
        optimizer.zero_grad()
        output = deep_sub[0](rand_batch)
        output = deep_sub[1](output)
        output = deep_sub[2](output, rand_batch)
        # true val calc
        x = gt_module[0](rand_batch)
        true_val = gt_module[1](x, rand_batch)
        loss = criterion(output, true_val)
        loss.backward()
        optimizer.step()
        pg_bar.set_description(f"Loss: {loss.item()}")
    return deep_sub


def train_deep_sub_svd(
    deep_sub, svd_module, gt_module, training_iter, input_size, batch_size=512, lr=0.001
):
    criterion = MSELoss()
    optimizer = Adam(deep_sub.parameters(), lr=lr)
    pg_bar =tqdm.tqdm(range(training_iter)) 
    for _ in pg_bar:
        rand_batch = torch.randn((batch_size, input_size))
        optimizer.zero_grad()
        output = deep_sub[0](rand_batch)
        output = deep_sub[1](output, rand_batch)

        svd_output = svd_module(rand_batch)

        # true val calc
        x = gt_module[0](rand_batch)
        true_val = gt_module[1](x, rand_batch)

        loss = criterion(output + svd_output, true_val)
        loss.backward()
        optimizer.step()
        pg_bar.set_description(f"Loss: {loss.item()}")
    return deep_sub


# core idea: for all the attention heads, keep track of the attention matrix score across
# some set of inputs.
# at the end, compute the fro norm of all the resulting matrices. For each layer, prune all but the top
# k attention heads
# if we change the dataframe / ex text, do the heads pruned change? do the top heads change? etc
def prune_attention_heads(bert_model, dataset, indices_to_use, save_dir, topk=1, strat="distortion"):
    if strat == "distortion":
        attention_matrices = compute_attention_matrices_distortion(bert_model, dataset, indices_to_use)
    else:
        attention_matrices = compute_attention_matrices(bert_model, dataset, indices_to_use)
    prune_dict = {}
    actual_frob_norms = {}
    for layer_idx in range(attention_matrices.size(0)):
        cur_layers = []
        for head_idx in range(attention_matrices.size(1)):
            frob_norm = torch.norm(
                attention_matrices[layer_idx, head_idx, :, :], p="fro"
            )
            cur_layers.append(frob_norm)

        sorted = np.argsort(np.array(cur_layers))
        prune_dict[layer_idx] = list([int(l) for l in sorted])[: len(sorted) - topk]
        actual_frob_norms[layer_idx] = list([float(l) for l in cur_layers])

    with open(f"{save_dir}/prune_dict.json", "w") as f:
        print(prune_dict)
        json.dump(prune_dict, f)
    with open(f"{save_dir}/frob_norms.json", "w") as f:
        # print(actual_frob_norms)
        json.dump(actual_frob_norms, f)
    print(bert_model.num_parameters())
    bert_model.prune_heads(prune_dict)
    print(bert_model.num_parameters())
    return bert_model


# function for computing summary attention matrix values
# this is a 4 dimension tensor:
# first dim = layer,
# second dim = head in layer,
# third dim = keys,
# fourth dim = queries
def compute_attention_matrices_distortion(bert_model, 
                               data_set, 
                               indices_to_use):
    running_attention_first_moment = None
    total = len(indices_to_use)
    with torch.no_grad():
        for i in tqdm.tqdm(indices_to_use):
            try: 
                x = data_set[i]
                rand_gibberish = data_set.random_gibberish(x["input_ids"])
                rand_masking = data_set.random_mask(x["input_ids"])
                # get the attention scores
                attention_natural = get_attention_scores(bert_model, x)
                x["input_ids"] = rand_gibberish
                attention_gibberish = get_attention_scores(bert_model, x)
                x["input_ids"] = rand_masking
                attention_masking = get_attention_scores(bert_model, x)

                attention_diff = (attention_natural - attention_gibberish) + (attention_natural - attention_masking)
                
                if running_attention_first_moment == None:
                    running_attention_first_moment = attention_diff
                else:
                    running_attention_first_moment += attention_diff
            except Exception as e:
                print(e)
                continue
    # var[X] = (E[X])^2 - E[X^2]
    return running_attention_first_moment / total


def compute_attention_matrices(bert_model, 
                               data_set, 
                               indices_to_use):
    running_attention_first_moment = None
    running_attention_second_moment = None 
    total = len(indices_to_use)
    try: 
        with torch.no_grad():
            for i in tqdm.tqdm(indices_to_use):
                natural_text = data_set[i]
                # get the attention scores
                attention = get_attention_scores(bert_model, natural_text)
                if running_attention_first_moment == None:
                    running_attention_first_moment = attention
                else:
                    running_attention_first_moment += attention

                if running_attention_second_moment == None:
                    running_attention_second_moment = attention**2
                else: 
                    running_attention_second_moment += attention**2
    except Exception as e: 
        print(e)
    # var[X] = (E[X])^2 - E[X^2]
    first_moment = running_attention_first_moment / total
    second_moment = running_attention_second_moment / total

    var = first_moment**2 - second_moment
    return var



def get_attention_scores(bert_model, x):
    # tokenize the input
    # tokenized_input = bert_model.tokenizer(text, return_tensors="pt")
    out = (
        torch.stack(
            bert_model(
                input_ids=x["input_ids"],
                token_type_ids=x["token_type_ids"],
                attention_mask=x["attention_mask"],
            ).attentions
        )
        .permute(1, 0, 2, 3, 4)
        .squeeze()
    )
    return out



def get_scaling_info_inputs(bert_model, dataset, sample_indices): 
    def caching_hook(module, input, output):
        module.input = input 
    
    hook_handle = reg_hook(caching_hook)
    loss = nn.CrossEntropyLoss()
    intermediate_first_moment = [None for _ in range(len(bert_model.bert.encoder.layer))]
    output_first_moment = [None for _ in range(len(bert_model.bert.encoder.layer))]
    with torch.no_grad():
        for i in tqdm.tqdm(sample_indices):
            # just for zeroing out the gradients
            x = dataset[i]
            bert_model(
                    input_ids=x["input_ids"],
                    token_type_ids=x["token_type_ids"],
                    attention_mask=x["attention_mask"],
                )
            for layer_idx, layer in enumerate(bert_model.bert.encoder.layer):
                intermediate = layer.intermediate.dense.input[0].squeeze(0)
                if intermediate_first_moment[layer_idx] == None:
                    intermediate_first_moment[layer_idx] = intermediate
                else:
                    intermediate_first_moment[layer_idx] += intermediate
                
                output = layer.output.dense.input[0].squeeze(0)
                if output_first_moment[layer_idx] == None:
                    output_first_moment[layer_idx] = output
                else: 
                    output_first_moment[layer_idx] += output
    hook_handle.remove()
    return {
        "intermediate_first_moment": [x/len(dataset) for x in intermediate_first_moment],
        "output_first_moment": [x/len(dataset) for x in output_first_moment],
    }


def get_scaling_info_inputs_peturbed(bert_model, dataset, sample_indices): 
    def caching_hook(module, input, output):
        module.input = input 
    
    hook_handle = reg_hook(caching_hook)
    loss = nn.CrossEntropyLoss()
    intermediate_first_moment = [[None, None, None] for _ in range(len(bert_model.bert.encoder.layer))]
    output_first_moment = [[None, None, None] for _ in range(len(bert_model.bert.encoder.layer))]
    with torch.no_grad():
        for i in tqdm.tqdm(sample_indices):
            try: 
                # just for zeroing out the gradients
                x = dataset[i]

                x_natural = x["input_ids"]
                x_gibberish = dataset.random_gibberish(x_natural)
                x_masking = dataset.random_mask(x_natural)
                inputs_to_eval = [x_masking, x_gibberish, x_natural]
                for input_idx, input_type in enumerate(inputs_to_eval):
                    
                # gt_label = torch.nn.functional.one_hot(x["labels"], num_classes=2)
                    bert_model(
                            input_ids=input_type,
                            token_type_ids=x["token_type_ids"],
                            attention_mask=x["attention_mask"],
                        )
                    for layer_idx, layer in enumerate(bert_model.bert.encoder.layer):
                        intermediate = layer.intermediate.dense.input[0].squeeze(0)
                        if intermediate_first_moment[layer_idx][input_idx] == None:
                            intermediate_first_moment[layer_idx][input_idx] = intermediate
                        else:
                            intermediate_first_moment[layer_idx][input_idx] += intermediate
                        
                        output = layer.output.dense.input[0].squeeze(0)
                        if output_first_moment[layer_idx][input_idx] == None:
                            output_first_moment[layer_idx][input_idx] = output
                        else: 
                            output_first_moment[layer_idx][input_idx] += output
            except Exception as e:
                print(e)
                continue
    hook_handle.remove()
    final_intermediate_mean = []
    final_output_mean = []
    for x in intermediate_first_moment:
        for x_t in x:
            x_t /= len(dataset)
        final_intermediate_mean.append(2 * x[0] - x[1] - x[2])
        
    for x in output_first_moment:
        for x_t in x:
            x_t /= len(dataset)
        final_output_mean.append(2 * x[0] - x[1] - x[2])
    return {
        "intermediate_first_moment": final_intermediate_mean,
        "output_first_moment": final_output_mean,
    }

def get_scaling_info_grad(bert_model, dataset, sample_indices):
    loss = nn.CrossEntropyLoss()
    intermediate_first_moment = [None for _ in range(len(bert_model.bert.encoder.layer))]
    intermediate_second_moment = [None for _ in range(len(bert_model.bert.encoder.layer))]
    output_first_moment = [None for _ in range(len(bert_model.bert.encoder.layer))]
    output_second_moment = [None for _ in range(len(bert_model.bert.encoder.layer))]

    optim = Adam(bert_model.parameters(), lr=0.001)
    for i in tqdm.tqdm(sample_indices):
        # just for zeroing out the gradients
        x = dataset[i]
        gt_label = x["labels"]
        # gt_label = torch.nn.functional.one_hot(x["labels"], num_classes=2)
        output = bert_model(
                input_ids=x["input_ids"],
                token_type_ids=x["token_type_ids"],
                attention_mask=x["attention_mask"],
            )
        loss_val = loss(output.logits.softmax(dim=-1), gt_label)
        loss_val.backward()
        for layer_idx, layer in enumerate(bert_model.bert.encoder.layer):
            intermediate = layer.intermediate.dense.weight.grad
            if intermediate_first_moment[layer_idx] == None:
                intermediate_first_moment[layer_idx] = intermediate
                intermediate_second_moment[layer_idx] = intermediate**2
            else:
                intermediate_first_moment[layer_idx] += intermediate
                intermediate_second_moment[layer_idx] += intermediate**2
            
            output_grad = layer.output.dense.weight.grad
            if output_first_moment[layer_idx] == None:
                output_first_moment[layer_idx] = output_grad
                output_second_moment[layer_idx] = output_grad**2
            else: 
                output_first_moment[layer_idx] += output_grad
                output_second_moment[layer_idx] += output_grad**2
    intermediate_first_moment = [x / len(dataset) for x in intermediate_first_moment]
    output_first_moment = [x / len(dataset) for x in output_first_moment]
    intermediate_second_moment = [x / len(dataset) for x in intermediate_second_moment]
    output_second_moment = [x / len(dataset) for x in output_second_moment]

    return {
        "intermediate_first_moment": intermediate_first_moment,
        "output_first_moment": output_first_moment,
        "intermediate_second_moment": intermediate_second_moment,
        "output_second_moment": output_second_moment
    }

# main idea: replace the ffn in the bert model with a deeper and narrow network
# we train this network to approximate the original ffn using random noise as input
# strategies:
# 1. vanilla_ffn: replace the ffn with a deeper and narrow network
# 2. svd_ffn: replace the ffn with a svd compressed network -- no training of network
# 3. svd_ffn_train: replace the ffn with a svd compressed network and train the network
# 4. sparse_ffn: replace the ffn with a sparse network -- no training of network
def replace_bert_ffn(
    bert_model,
    strategy="vanilla_ffn",
    train_iter=1000,
    save_dir="substitute_networks",
    inner_rank=200,
    percentile=0.25,
    svd_bias=True,
    dataset = None, 
    grad_scaling = "none",
    sample_indices=None
):
    scaling_info = None 
    if grad_scaling == "grad": 
        scaling_info = get_scaling_info_grad(bert_model, dataset, sample_indices)
    elif grad_scaling == "inputs": 
        scaling_info = get_scaling_info_inputs(bert_model, dataset, sample_indices)
    elif grad_scaling == "inputs_peturbed":
        scaling_info = get_scaling_info_inputs_peturbed(bert_model, dataset, sample_indices)
    for layer_counter, cur_layer in enumerate(bert_model.bert.encoder.layer):
        intermediate_weights = cur_layer.intermediate.dense.weight.data
        output_weights = cur_layer.output.dense.weight.data

        sparse_intermediate = get_sparse_weights(intermediate_weights, percentile)
        sparse_output = get_sparse_weights(output_weights, percentile)

        cur_layer.intermediate.dense.weight.data = sparse_intermediate
        cur_layer.output.dense.weight.data = sparse_output

        gt_module = nn.Sequential(cur_layer.intermediate, cur_layer.output)

        if strategy == "vanilla_ffn":
            deep_sub_intermediate = DeepSub(
                input_dim=intermediate_weights.size(1),
                inner_rank=inner_rank,
                output_dim=inner_rank,
            )
            deep_sub_output = DeepSubOutput(
                input_dim=inner_rank,
                inner_rank=inner_rank,
                output_dim=intermediate_weights.size(1),
            )
            deep_sub_module = nn.Sequential(
                deep_sub_intermediate, nn.GELU(), deep_sub_output
            )
            deep_sub_module = train_deep_sub(
                deep_sub_module, gt_module, train_iter, intermediate_weights.size(1)
            )
            # replacing the big ffn with the deep sub module
            cur_layer.intermediate.dense = deep_sub_module[0]
            cur_layer.output = deep_sub_module[2]

        elif strategy == "svd_ffn":
            svd_module = get_svd_ffn(sparse_intermediate, 
                                     sparse_output, 
                                     inner_rank,
                                     scaling_info=scaling_info,
                                     layer_idx=layer_counter,
                                     scaling_type=grad_scaling)
            cur_layer.intermediate.dense = nn.Sequential(svd_module[0], svd_module[1])
            cur_layer.output.dense = nn.Sequential(svd_module[2], svd_module[3])

        elif strategy == "svd_ffn_train":
            svd_module = get_svd_ffn(
                intermediate_weights, output_weights, inner_rank, bias=svd_bias
            )

            svd_module = train_deep_sub(
                svd_module, gt_module, train_iter, intermediate_weights.size(1)
            )

            cur_layer.intermediate.dense = nn.Sequential(svd_module[0], svd_module[1])
            cur_layer.output.dense = nn.Sequential(svd_module[2], svd_module[3])

        elif strategy == "sparse_ffn":
            pass  # do nothing; already replaced the data with sparsified matrix
    bert_model.save_pretrained(f"{save_dir}")
    print("Saved Model")
    return bert_model
