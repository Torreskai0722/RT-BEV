import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV
data = pd.read_csv('camera_sweep_results.csv')

# Display the first few rows of the dataframe to understand its structure
print(data.head())

# Set style for seaborn
sns.set(style="whitegrid")

# Plot Processing Times by Crop Size for different numbers of cameras
plt.figure(figsize=(12, 6))
for i in data['Number of Cameras'].unique():
    subset = data[data['Number of Cameras'] == i]
    plt.plot(subset['Crop Width'], subset['Prev Frame Extraction Time'], marker='o', label=f'{i} Cameras - Prev Frame')
    plt.plot(subset['Crop Width'], subset['ROI Processing Time'], marker='o', linestyle='--', label=f'{i} Cameras - ROI')

plt.title('Processing Times vs. Crop Width for Different Numbers of Cameras')
plt.xlabel('Crop Width (pixels)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.savefig('Processing_Times_vs_Crop_Width.pdf')  # Save the figure to a file
plt.show()

# Plot Cosine Similarity and Euclidean Distance by Crop Size
plt.figure(figsize=(12, 6))
sns.lineplot(x='Crop Width', y='Cosine Similarity', hue='Number of Cameras', data=data, marker='o')
plt.title('Cosine Similarity by Crop Width for Different Numbers of Cameras')
plt.xlabel('Crop Width (pixels)')
plt.ylabel('Cosine Similarity')
plt.legend(title='Number of Cameras')
plt.savefig('Cosine_Similarity_vs_Crop_Width.pdf')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Crop Width', y='Euclidean Distance', hue='Number of Cameras', data=data, marker='o')
plt.title('Euclidean Distance by Crop Width for Different Numbers of Cameras')
plt.xlabel('Crop Width (pixels)')
plt.ylabel('Euclidean Distance')
plt.legend(title='Number of Cameras')
plt.savefig('Euclidean_Distance_vs_Crop_Width.pdf')
plt.show()
