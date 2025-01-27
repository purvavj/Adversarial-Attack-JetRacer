import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchattacks import FGSM
import matplotlib.pyplot as plt

# Data Augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomCrop(32, padding=4),  # Randomly crop and pad images
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 Dataset
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Model Definition
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

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = SimpleCNN().to(device)

# Training Configuration
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
n = 20  # Number of epochs
losses = []

# Training Loop
for epoch in range(n):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Generate adversarial examples
        fgsm = FGSM(model, eps=0.1)
        adv_images = fgsm(images, labels)
        
        # Combine clean and adversarial examples
        combined_images = torch.cat([images, adv_images])
        combined_labels = torch.cat([labels, labels])
        
        # Forward pass
        outputs = model(combined_images)
        loss = criterion(outputs, combined_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Accuracy on clean data
        clean_outputs = model(images)
        _, predicted = clean_outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{n}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

print("Model trained successfully!")

# Plot Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, n + 1), losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

# FGSM Evaluation Function
def evaluate_fgsm_for_epsilons(model, data_loader, epsilons):
    results = {"baseline": None, "fgsm": {}}
    model.eval()
    
    # Baseline accuracy
    correct, total = 0, 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    baseline_acc = 100 * correct / total
    results["baseline"] = baseline_acc
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    
    # FGSM accuracy for each epsilon
    for eps in epsilons:
        fgsm = FGSM(model, eps=eps)
        correct, total = 0, 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            adv_images = fgsm(images, labels)
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        fgsm_acc = 100 * correct / total
        results["fgsm"][eps] = fgsm_acc
        print(f"Epsilon: {eps}, Accuracy after FGSM attack: {fgsm_acc:.2f}%")
    
    return results

# FGSM Results
epsilons = [0.1, 0.3, 0.5]
fgsm_results = evaluate_fgsm_for_epsilons(model, test_loader, epsilons)

# Check Adversarial Perturbations
def check_perturbations(images, adv_images):
    perturbations = adv_images - images
    print(f"Mean perturbation: {perturbations.abs().mean().item()}")
    print(f"Max perturbation: {perturbations.abs().max().item()}")

data_iter = iter(test_loader)
images, labels = next(data_iter)
images, labels = images.to(device), labels.to(device)
fgsm = FGSM(model, eps=0.3)
adv_images = fgsm(images, labels)
check_perturbations(images, adv_images)