from torch.nn.functional import cosine_similarity

def euclidean_distance(x, y):
    return (x - y).norm(p=2, dim=1)  # Compute L2 norm across the dimension 1

def calculate_sim(previous_results, final_outputs):
    pre_output = previous_results.permute(1, 0, 2).squeeze(0)
    curr_output = final_outputs.permute(1, 0, 2).squeeze(0)

    print(pre_output.shape, curr_output.shape)
    # Cosine similarity for each pair in the batch
    cos_sim = cosine_similarity(curr_output, pre_output)  # Output shape: [6]
    print("Cosine Similarities:", cos_sim)

    # Calculate Euclidean distances
    distances = euclidean_distance(curr_output, pre_output)
    print("Euclidean Distances:", distances)

    # Average cosine similarity across the batch
    average_cos_sim = cosine_similarity(curr_output.view(1, -1), pre_output.view(1, -1))

    # Average Euclidean distance across the batch
    average_distance = euclidean_distance(curr_output.view(1, -1), pre_output.view(1, -1))

    average_cos_sim = average_cos_sim.cpu().detach().numpy().item()
    average_distance = average_distance.cpu().detach().numpy().item()

    print("Average Cosine Similarity:", average_cos_sim)
    print("Average Euclidean Distance:", average_distance)

    return average_cos_sim, average_distance