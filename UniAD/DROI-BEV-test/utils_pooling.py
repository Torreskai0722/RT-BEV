import torch

# Assume 'outputs' is your (24, 1000) tensor
# outputs = torch.rand(24, 1000)  # Example data

def average_pooling(outputs, num_groups):
    # Reshape outputs to (num_groups, -1, feature_dim) where -1 infers dimension to fit all elements
    outputs = outputs.view(num_groups, -1, outputs.shape[-1])
    # Compute mean across the second dimension
    pooled_outputs = outputs.mean(dim=1)
    return pooled_outputs

def max_pooling(outputs, num_groups):
    outputs = outputs.view(num_groups, -1, outputs.shape[-1])
    pooled_outputs = outputs.max(dim=1)[0]  # [0] to get values only, [1] would give indices
    return pooled_outputs

def sum_pooling(outputs, num_groups):
    outputs = outputs.view(num_groups, -1, outputs.shape[-1])
    pooled_outputs = outputs.sum(dim=1)
    return pooled_outputs

def weighted_pooling(outputs, num_groups, weights):
    outputs = outputs.view(num_groups, -1, outputs.shape[-1])
    # Ensure weights sum to 1 for each group
    weights = torch.FloatTensor(weights).view(num_groups, -1, 1)
    weights /= weights.sum(dim=1, keepdim=True)
    pooled_outputs = (outputs * weights).sum(dim=1)
    return pooled_outputs