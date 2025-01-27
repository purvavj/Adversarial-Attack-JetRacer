import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchattacks import PGD
import matplotlib.pyplot as plt

# Step 1: Data Augmentation and Loading
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Step 2: Define the Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = SimpleCNN().to(device)

# Step 3: Train the Model (Reuse Training Code)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n = 20  # Number of epochs
for epoch in range(n):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pgd = PGD(model, eps=0.3, alpha=0.01, steps=40)
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        adv_images = pgd(images, labels)
        
        # Combine clean and adversarial examples
        combined_images = torch.cat([images, adv_images])
        combined_labels = torch.cat([labels, labels])
        
        optimizer.zero_grad()
        outputs = model(combined_images)
        loss = criterion(outputs, combined_labels)
        loss.backward()
        optimizer.step()

        # Update loss
        running_loss += loss.item()

        # Update accuracy
        _, predicted = outputs.max(1)  # Get predictions
        correct += (predicted == combined_labels).sum().item()  # Count correct
        total += combined_labels.size(0)  # Update total

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total  # Now `total` is updated
    print(f"Epoch [{epoch + 1}/{n}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

print("Model trained successfully!")

# Step 4: Evaluate the Model Against PGD
pgd = PGD(model, eps=0.3, alpha=0.01, steps=40)  # Initialize PGD attack

def evaluate_model_with_attack(model, data_loader, attack=None):
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

# Test the Model on Clean Images
clean_accuracy = evaluate_model_with_attack(model, test_loader)
print(f"Baseline Accuracy on clean images: {clean_accuracy:.2f}%")

# Test the Model with PGD Attack
pgd_accuracy = evaluate_model_with_attack(model, test_loader, attack=pgd)
print(f"Accuracy after PGD attack: {pgd_accuracy:.2f}%")