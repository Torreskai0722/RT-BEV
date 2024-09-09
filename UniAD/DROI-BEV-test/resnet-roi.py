import time
import torch
import torchvision.transforms.functional as TF
import torchvision.models as models
import random
from similarity_tensor import calculate_sim

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize the ResNet-101 model, modify it, and move to GPU
resnet101 = models.resnet101(pretrained=True).to(device)
modules = list(resnet101.children())[:-2]  # Remove the average pool and fc layer
resnet101 = torch.nn.Sequential(*modules).to(device)

# Reattach the average pooling and FC layer, moved to GPU
avg_pool = models.resnet101(pretrained=True).avgpool.to(device)
fc_layer = models.resnet101(pretrained=True).fc.to(device)

# Simulated batch of frames from 6 cameras
frames = torch.rand(6, 3, 900, 1600, device=device)  # Create the tensor directly on the GPU

with torch.no_grad():
    torch.cuda.synchronize(device)
    t0 = time.time()
    previous_features = resnet101(frames)
    # Apply average pooling and FC layer
    pre_pooled_outputs = [avg_pool(previous_features[i].unsqueeze(0)) for i in range(6)]
    pre_pooled_outputs = torch.flatten(torch.stack(pre_pooled_outputs), 1)
    previous_results = torch.stack([fc_layer(pre_pooled_outputs[i].unsqueeze(0)) for i in range(pre_pooled_outputs.size(0))])
    torch.cuda.synchronize(device)
    t1 = time.time()

flush_frame = torch.rand(6, 3, 300, 1000, device=device)  # Create the tensor directly on the GPU
with torch.no_grad():
    _ = resnet101(flush_frame)

# Define how many cameras to select randomly
num_selected_cameras = 1  # For example, select 3 cameras randomly

# Generate random indices for the cameras
selected_indices = random.sample(range(frames.size(0)), num_selected_cameras)
print(selected_indices)

crop_h = 224
crop_w = 224

# Define RoIs for each selected camera
rois = [(random.randint(0, 1600-crop_w), random.randint(0, 900-crop_h), crop_w, crop_h) for _ in range(num_selected_cameras)]

# Crop the RoIs from each selected frame
roi_frames = torch.stack([TF.crop(frames[i], rois[idx][1], rois[idx][0], rois[idx][2], rois[idx][3]) for idx, i in enumerate(selected_indices)])

# Forward the RoIs through the modified network to get feature maps
with torch.no_grad():
    # torch.cuda.synchronize(device)
        
    torch.cuda.synchronize(device)
    t2 = time.time()
    roi_features = resnet101(roi_frames)  # Output shape should be [num_selected_cameras, 2048, 7, 7]
    torch.cuda.synchronize(device)
    t3 = time.time()

torch.cuda.synchronize(device)
t4 = time.time()
# Replace the corresponding sections in the previous feature maps
for idx, camera_idx in enumerate(selected_indices):
    center_x, center_y = rois[idx][0] // 64, rois[idx][1] // 60  # Adjust for input size to feature map size
    start_x = max(center_x - 7 // 2, 0)
    start_y = max(center_y - 7 // 2, 0)
    previous_features[camera_idx, :, start_y:start_y + 7, start_x:start_x + 7] = roi_features[idx]

torch.cuda.synchronize(device)
t5 = time.time()

torch.cuda.synchronize(device)
t6 = time.time()
# Apply average pooling and FC layer
pooled_outputs = [avg_pool(previous_features[i].unsqueeze(0)) for i in selected_indices]
pooled_outputs = torch.flatten(torch.stack(pooled_outputs), 1)
outputs = torch.stack([fc_layer(pooled_outputs[i].unsqueeze(0)) for i in range(pooled_outputs.size(0))])

print(outputs.shape)
final_outputs = previous_results.clone()

if len(selected_indices) == 1:
    final_outputs[selected_indices] = outputs
else:
    # Update the outputs only for selected cameras
    for index, data in enumerate(selected_indices):
        final_outputs[data] = outputs[index]

# Check the output shapes
print("Final output shapes:", final_outputs.shape)  # Should be [num_selected_cameras, num_classes]
torch.cuda.synchronize(device)
t7 = time.time()

print("Previous frame feature extraction time:", t1 - t0)
print("Current frame processing times:")
print("ROI processing time:", t3 - t2)
print("Pooling and FC time:", t4-t3, t5-t4, t6-t5, t7-t6, t7-t3)
print("Overall processing time since first inference:", t7 - t2)

calculate_sim(previous_results, final_outputs)
