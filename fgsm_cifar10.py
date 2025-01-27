import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchattacks 
import matplotlib.pyplot as plt


# Data Augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomCrop(32, padding=4),  # Randomly crop and pad images
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Load CIFAR-10 Dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Updated Dataset
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

# Model training
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n = 40 #no of epoch
for epoch in range(n):  
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
    print(f"Epoch [{epoch + 1}/{n}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

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
atk = torchattacks.FGSM(model, eps=0.1)
fgsm_accuracy = evaluate_model(model, test_loader, attack=atk)
print(f"Accuracy after FGSM attack: {fgsm_accuracy:.2f}%")

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
        plt.imshow((images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).numpy())  # Unnormalize
        plt.title(f"Original: {labels[i].item()}")
        plt.axis("off")

        # Adversarial image
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow((adv_images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).numpy())  # Unnormalize
        plt.title("Adversarial")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

visualize_adversarial_examples(model, test_loader, atk)