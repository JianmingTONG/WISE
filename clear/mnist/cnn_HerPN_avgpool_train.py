#!/home/jyh/project/mnist/mnist-venv/bin/python3
# run  := python3 cnn_HerPN_avgpool_train.py
# dir  := .
# kid  :=

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from HerPN import HerPN

# 1. Hyperparameters and Device Configuration
# Try to use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 64
learning_rate = 0.001
num_epochs = 10 # You can increase this for better accuracy
random_seed = 42 # For reproducibility

# Set random seed for reproducibility
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

# 2. MNIST Dataset Loading and Preprocessing
# Define transformations
# ToTensor() converts a PIL Image or numpy.ndarray (H x W x C) in the range
# [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
# Normalize() normalizes a tensor image with mean and standard deviation.
# For MNIST, the mean and standard deviation are widely known as 0.1307 and 0.3081 respectively.
# We will pad the 28x28 image to 32x32. Pad with 2 pixels on each side (28+2+2=32).
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Pad(2), # Pads to 32x32
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the training data
train_dataset = datasets.MNIST(root='../data',
                               train=True,
                               transform=transform,
                               download=True)

# Download and load the test data
test_dataset = datasets.MNIST(root='../data',
                              train=False,
                              transform=transform,
                              download=True)

# Create DataLoaders
# DataLoader provides an iterable over the given dataset.
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False) # No need to shuffle test data

# Let's check one batch of images
examples = iter(train_loader)
example_data, example_targets = next(examples)
print(f"Example data shape: {example_data.shape}") # Should be [batch_size, 1, 28, 28]
print(f"Example targets shape: {example_targets.shape}") # Should be [batch_size]

H_last = W_last = 7

# 3. Model Definition (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional Layer 1
        # Input channels = 1 (grayscale image)
        # Output channels = 32
        # Kernel size = 3x3
        # Padding = 1 (to keep the image size the same: (28-3+2*1)/1 + 1 = 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.act1 = HerPN(num_features=32, degree=2) # Example usage of HerPN
        # Max Pooling Layer 1
        # Kernel size = 2x2
        # Stride = 2 (halves the image dimensions: 28x28 -> 14x14)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2
        # Input channels = 32 (from conv1)
        # Output channels = 64
        # Kernel size = 3x3
        # Padding = 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.act2 = HerPN(num_features=64, degree=2) # Example usage of HerPN
        # Max Pooling Layer 2
        # Kernel size = 2x2
        # Stride = 2 (halves the image dimensions: 14x14 -> 7x7)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer 1
        # Input features: 64 channels * 7x7 image size = 64 * 49 = 3136
        # Output features = 128 (arbitrary choice, can be tuned)
        self.fc1 = nn.Linear(64 * H_last * W_last, 196)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=0)
        self.act3 = HerPN(num_features=196, degree=2) # Example usage of HerPN
        # Dropout layer with p=0.5 (50% of neurons are randomly zeroed out during training)
        self.dropout = nn.Dropout(0.5)
        # Fully Connected Layer 2 (Output Layer)
        # Input features = 128 (from fc1)
        # Output features = 10 (number of classes, digits 0-9)
        self.fc2 = nn.Linear(196, 10)

    def forward(self, x):
        # x shape: [batch_size, 1, 32, 32]
        x = self.conv1(x) # [batch_size, 32, 32, 32]
        x = self.act1(x)    # Activation
        x = self.pool1(x) # [batch_size, 32, 16, 16]

        x = self.conv2(x) # [batch_size, 64, 16, 16]
        x = self.act2(x)     # Activation
        x = self.pool2(x) # [batch_size, 64, 8, 8]

        # Flatten the tensor for the fully connected layer
        # x.view(-1, num_features)
        # -1 means infer the size from other dimensions (i.e., batch_size)
        x = x.view(-1, 64 * H_last * W_last) # [batch_size, 4096]

        x = self.fc1(x)   # [batch_size, 128]
        x = self.act3(x)     # Activation
        x = self.dropout(x) # Apply dropout

        x = self.fc2(x)   # [batch_size, 10] (raw scores/logits)
        # No softmax here because nn.CrossEntropyLoss applies it internally.
        # If using NLLLoss, you would apply F.log_softmax(x, dim=1) here.
        return x


# Instantiate the model and move it to the configured device
model = CNN().to(device)
# print(model) # To see the model architecture

# 4. Loss Function and Optimizer
criterion = nn.CrossEntropyLoss() # Combines LogSoftmax and NLLLoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. Training Loop
def train_model(num_epochs):
    model.train() # Set the model to training mode
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad() # Clear previous gradients
            loss.backward()       # Compute gradients
            optimizer.step()        # Update weights

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {running_loss/total_steps:.4f}')
    print('Finished Training')
    torch.save(model.state_dict(), "./mnist_cnn_herpn_avgpool_wo_padding_196.pth")

# 6. Evaluation Loop
def evaluate_model():
    model.eval() # Set the model to evaluation mode
    # In evaluation phase, we don't need to compute gradients
    with torch.no_grad():
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # torch.max returns (value, index)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the {total} test images: {accuracy:.2f} %')
        return accuracy, all_preds, all_labels

# 7. Running the training and evaluation
print("Starting training...")
train_model(num_epochs)
print("\nStarting evaluation...")
accuracy, predictions, true_labels = evaluate_model()

# 8. Displaying some predictions (optional)
def plot_predictions(images, labels, preds, num_images=10):
    plt.figure(figsize=(12, 12))
    # Get a batch of test images
    dataiter = iter(test_loader)
    sample_images, sample_labels = next(dataiter)

    # Denormalize images for display
    # Mean and std used for normalization
    mean = 0.1307
    std = 0.3081
    denormalized_images = []
    for i in range(num_images):
        img = sample_images[i].squeeze().cpu().numpy() # Squeeze to remove channel dim, move to CPU, convert to numpy
        img = (img * std) + mean # Denormalize
        img = np.clip(img, 0, 1) # Clip values to be between 0 and 1
        denormalized_images.append(img)

    # Make predictions for these sample images
    sample_images_tensor = sample_images[:num_images].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(sample_images_tensor)
        _, predicted_classes = torch.max(outputs, 1)
    predicted_classes = predicted_classes.cpu().numpy()

    for i in range(num_images):
        plt.subplot(5, 2, i + 1)
        plt.imshow(denormalized_images[i], cmap='gray')
        plt.title(f"True: {sample_labels[i].item()}, Pred: {predicted_classes[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

# print("\nPlotting some predictions...")
# Get a batch of test images for plotting (uses the test_loader logic)
# plot_predictions(test_loader, true_labels, predictions, num_images=10)

# To save the model (optional)
# torch.save(model.state_dict(), 'mnist_cnn_model.pth')
# print("Model saved to mnist_cnn_model.pth")

# To load a pre-trained model (optional)
# model_loaded = CNN().to(device)
# model_loaded.load_state_dict(torch.load('mnist_cnn_model.pth'))
# model_loaded.eval() # Don't forget to set to eval mode
# accuracy_loaded, _, _ = evaluate_model_with_instance(model_loaded, test_loader) # You'd need to adapt evaluate_model
# print(f'Accuracy of the loaded model: {accuracy_loaded:.2f} %')
