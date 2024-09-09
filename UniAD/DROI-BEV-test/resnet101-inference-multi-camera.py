import time
import torch
import torchvision.transforms.functional as TF
import torchvision.models as models
import random
from similarity_tensor import calculate_sim
from itertools import product
import csv

def process_camera_frames(num_selected_cameras, crop_x, crop_y, writer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize the ResNet-101 model, modify it, and move to GPU
    resnet101 = models.resnet101(pretrained=True).to(device)
    modules = list(resnet101.children())[:-2]
    resnet101 = torch.nn.Sequential(*modules).to(device)
    avg_pool = models.resnet101(pretrained=True).avgpool.to(device)
    fc_layer = models.resnet101(pretrained=True).fc.to(device)

    # Simulated batch of frames from 6 cameras
    frames = torch.rand(6, 3, 900, 1600, device=device)

    # Process previous frames to establish a baseline for comparison
    with torch.no_grad():
        torch.cuda.synchronize(device)
        t0 = time.time()
        previous_features = resnet101(frames)
        pre_pooled_outputs = [avg_pool(previous_features[i].unsqueeze(0)) for i in range(6)]
        pre_pooled_outputs = torch.flatten(torch.stack(pre_pooled_outputs), 1)
        previous_results = torch.stack([fc_layer(pre_pooled_outputs[i].unsqueeze(0)) for i in range(pre_pooled_outputs.size(0))])
        torch.cuda.synchronize(device)
        t1 = time.time()

    flush_frame = torch.rand(6, 3, 300, 1000, device=device)  # Create the tensor directly on the GPU
    with torch.no_grad():
        _ = resnet101(flush_frame)

    # Generate random indices for the cameras
    selected_indices = random.sample(range(frames.size(0)), num_selected_cameras)
    print(selected_indices)

    # Define RoIs for each selected camera
    rois = [(random.randint(0, 1600-crop_x), random.randint(0, 900-crop_y), crop_x, crop_y) for _ in range(num_selected_cameras)]

    # Crop the RoIs from each selected frame
    roi_frames = torch.stack([TF.crop(frames[i], rois[idx][1], rois[idx][0], rois[idx][2], rois[idx][3]) for idx, i in enumerate(selected_indices)])

    # Forward the RoIs through the modified network to get feature maps
    with torch.no_grad():
        torch.cuda.synchronize(device)
        t2 = time.time()
        roi_features = resnet101(roi_frames)
        torch.cuda.synchronize(device)
        t3 = time.time()

    # Dynamically adjust the update region to match the actual sizes
    for idx, camera_idx in enumerate(selected_indices):
        target_shape = roi_features.shape[2:]  # Get the actual shape of the output feature map
        previous_features[camera_idx, :, :target_shape[0], :target_shape[1]] = roi_features[idx]

    # # Replace the corresponding sections in the previous feature maps
    # for idx, camera_idx in enumerate(selected_indices):
    #     center_x, center_y = rois[idx][0] // 64, rois[idx][1] // 60
    #     start_x = max(center_x - 7 // 2, 0)
    #     start_y = max(center_y - 7 // 2, 0)
    #     previous_features[camera_idx, :, start_y:start_y + 7, start_x:start_x + 7] = roi_features[idx]

    # Apply average pooling and FC layer
    pooled_outputs = [avg_pool(previous_features[i].unsqueeze(0)) for i in selected_indices]
    pooled_outputs = torch.flatten(torch.stack(pooled_outputs), 1)
    outputs = torch.stack([fc_layer(pooled_outputs[i].unsqueeze(0)) for i in range(pooled_outputs.size(0))])

    # Update the final outputs tensor with new data
    final_outputs = previous_results.clone()
    if len(selected_indices) == 1:
        final_outputs[selected_indices] = outputs
    else:
        for index, data in enumerate(selected_indices):
            final_outputs[data] = outputs[index]

    torch.cuda.synchronize(device)
    t7 = time.time()

    print("Final output shapes:", final_outputs.shape)
    print("Previous frame feature extraction time:", t1 - t0)
    print("Current frame processing times:")
    print("ROI processing time:", t3 - t2)
    print("Overall processing time since first inference:", t7 - t2)

    # Calculate similarity
    ave_cos, ave_eli = calculate_sim(previous_results, final_outputs)

    # Write results to CSV
    writer.writerow([num_selected_cameras, crop_x, crop_y, t1-t0, t3-t2, ave_cos, ave_eli])

    print("Processed for", num_selected_cameras, "cameras with crop size", crop_x, crop_y)


def sweep_camera_settings():
    with open('camera_sweep_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Number of Cameras', 'Crop Width', 'Crop Height', 'Prev Frame Extraction Time', 'ROI Processing Time', 'Cosine Similarity', 'Euclidean Distance'])

        # Parameter ranges
        num_cameras_options = range(1, 7)  # From 1 to 6 cameras
        crop_sizes = range(224, 801, 32)  # From 224 to 800 with a step of 32

        # Sweep through all combinations
        for num_cameras, crop_size in product(num_cameras_options, crop_sizes):
            process_camera_frames(num_cameras, crop_size, crop_size, writer)

# Run the sweep
sweep_results = sweep_camera_settings()

# Example usage
# num_cameras = 3
# crop_width = 224
# crop_height = 224
# similarity = process_camera_frames(num_cameras, crop_width, crop_height)