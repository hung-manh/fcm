import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
import skfuzzy as fuzz
from ipywidgets import interact, widgets
from IPython.display import display

# Load the demo image from a public URL
url = 'https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png'
image = io.imread(url)

# Handle images with an alpha channel by considering only the first three channels (RGB)
if image.shape[-1] == 4:
    image = image[..., :3]

# Convert the image to grayscale
gray_image = rgb2gray(image)

# Display the original and grayscale images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')
plt.show()

# Reshape the image into a 1D array
pixels = gray_image.reshape(-1, 1)

# Define the number of clusters
n_clusters = 3

# Apply Fuzzy C-Means clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    pixels.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)

# Get the cluster membership for each pixel
cluster_membership = np.argmax(u, axis=0)

# Reshape the clustered data back to the original image shape
segmented_image = cluster_membership.reshape(gray_image.shape)

# Display the segmented image
plt.figure(figsize=(8, 8))
plt.title('Segmented Image')
plt.imshow(segmented_image, cmap='gray')
plt.show()

# Create an empty image with the same shape as the original
colored_segmented_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3))

# Assign colors to each cluster
for i in range(n_clusters):
    colored_segmented_image[segmented_image == i] = np.random.rand(3)

# Display the color-segmented image
plt.figure(figsize=(8, 8))
plt.title('Color-Segmented Image')
plt.imshow(colored_segmented_image)
plt.show()

# Save the segmented image
io.imsave('segmented_image.jpg', (colored_segmented_image * 255).astype(np.uint8))

def segment_image(url, n_clusters):
    try:
        # Load the image from the provided URL
        image = io.imread(url)

        # Handle images with an alpha channel by considering only the first three channels (RGB)
        if image.shape[-1] == 4:
            image = image[..., :3]

        # Convert the image to grayscale
        gray_image = rgb2gray(image)

        # Reshape the image into a 1D array
        pixels = gray_image.reshape(-1, 1)

        # Apply Fuzzy C-Means clustering
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            pixels.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)

        # Get the cluster membership for each pixel
        cluster_membership = np.argmax(u, axis=0)

        # Reshape the clustered data back to the original image shape
        segmented_image = cluster_membership.reshape(gray_image.shape)

        # Create an empty image with the same shape as the original
        colored_segmented_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3))

        # Assign colors to each cluster
        for i in range(n_clusters):
            colored_segmented_image[segmented_image == i] = np.random.rand(3)

        # Display the original, grayscale, and color-segmented images
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.subplot(1, 3, 2)
        plt.title('Grayscale Image')
        plt.imshow(gray_image, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title('Color-Segmented Image')
        plt.imshow(colored_segmented_image)
        plt.show()
    except Exception as e:
        print(f"Error loading image from URL: {e}")

# Create the widgets
url_widget = widgets.Text(
    value='https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png',
    description='Image URL:',
    layout=widgets.Layout(width='80%')
)
clusters_widget = widgets.IntSlider(
    value=3,
    min=2,
    max=10,
    step=1,
    description='Clusters:'
)

# Use the interact function to create the GUI
interact(segment_image, url=url_widget, n_clusters=clusters_widget)