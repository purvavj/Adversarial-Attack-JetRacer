import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchattacks
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


# MNIST Dataset loads
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Function to show a batch of images
# def show_images(loader):
#     dataiter = iter(loader)
#     images, labels = next(dataiter)  # Get a batch
#     images = images[:16]  # Show first 16 images
#     grid = make_grid(images, nrow=4)  # Arrange in a grid
#     plt.figure(figsize=(8, 8))
#     plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
#     plt.title("Sample MNIST Images")
#     plt.axis("off")
#     plt.show()
# show_images(train_loader)

# Define a Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model training
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):  
    model.train()
    running_loss = 0.0  # Track cumulative loss for the epoch
    correct = 0         # Track correct predictions
    total = 0           # Track total predictions

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update loss
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Print metrics after each epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/20], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

print("Model trained successfully!")

def evaluate_model(model, data_loader, attack=None):
    model.eval()  
    correct = 0
    total = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        if attack:
            images = attack(images, labels)  

        outputs = model(images)
        _, predicted = outputs.max(1) 
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy

baseline_accuracy = evaluate_model(model, test_loader)
print(f"Baseline Accuracy on clean images: {baseline_accuracy:.2f}%")

# checking accuracy after FGSM Attack
atk = torchattacks.FGSM(model, eps=0.3)
fgsm_accuracy = evaluate_model(model, test_loader, attack=atk)
print(f"Accuracy after FGSM attack: {fgsm_accuracy:.2f}%")

# Visualize Adversarial Examples
def visualize_adversarial_examples(model, data_loader, attack, num_images=8):
    model.eval()
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images, labels = images[:num_images].to(device), labels[:num_images].to(device)

    # Generate adversarial examples
    adv_images = attack(images, labels)

    # Plot clean and adversarial images side-by-side
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        # Original image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].cpu().squeeze(), cmap="gray")
        plt.title(f"Original: {labels[i].item()}")
        plt.axis("off")

        # Adversarial image
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(adv_images[i].cpu().squeeze(), cmap="gray")
        plt.title("Adversarial")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

visualize_adversarial_examples(model, test_loader, atk)

