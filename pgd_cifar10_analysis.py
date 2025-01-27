import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchattacks import PGD
import matplotlib.pyplot as plt

from pgd_cifar10 import evaluate_model_with_attack

# Data Augmentation
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

# Define the Model
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

# Training Configuration
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

n = 30  # Total epochs
epsilons = [0.1, 0.3, 0.5]  # Curriculum progression for adversarial training
current_eps = epsilons[0]  # Start with the first epsilon

for epoch in range(n):
    # Update epsilon based on curriculum
    if epoch < 10:
        current_eps = epsilons[0]
    elif epoch < 20:
        current_eps = epsilons[1]
    else:
        current_eps = epsilons[2]

    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    pgd = PGD(model, eps=current_eps, alpha=0.01, steps=40)

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd(images, labels)

        # Combine clean and adversarial examples
        combined_images = torch.cat([images, adv_images])
        combined_labels = torch.cat([labels, labels])

        optimizer.zero_grad()
        outputs = model(combined_images)
        loss = criterion(outputs, combined_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == combined_labels).sum().item()
        total += combined_labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{n}], Epsilon: {current_eps}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    scheduler.step()

print("Model trained successfully with curriculum adversarial training!")
torch.save(model.state_dict(), "curriculum_pgd_model.pth")
print("Saved model as 'curriculum_pgd_model.pth'.")

# Evaluate the Model
pgd_attack = PGD(model, eps=0.3, alpha=0.01, steps=40)
pgd_accuracy = evaluate_model_with_attack(model, test_loader, attack=pgd_attack)
print(f"Accuracy after PGD attack (eps=0.3): {pgd_accuracy:.2f}%")