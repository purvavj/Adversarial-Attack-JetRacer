import torch
from torchattacks import PGD
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import os


from pgd_cifar10 import SimpleCNN  

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Custom Dataset for KITTI
class KITTIDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = 0  # Dummy label since KITTI does not have labeled classes for classification
        if self.transform:
            image = self.transform(image)
        return image, label


# Load KITTI Dataset
def load_kitti_data(data_dir, batch_size=16):
    transform = Compose([
        Resize((64, 64)),  # Resize images to fit the model input
        ToTensor(),
        Normalize((0.5,), (0.5,))  # Normalize based on expected model input
    ])
    dataset = KITTIDataset(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)



kitti_data_dir = "./kitti_data/data_object_image_2/training/image_2"  # Update to your KITTI dataset path
kitti_loader = load_kitti_data(kitti_data_dir)


model = SimpleCNN().to(device)
model.load_state_dict(torch.load("curriculum_pgd_model.pth"))
model.eval()


# Targeted Attack
def targeted_attack(model, data_loader, target_class, eps=0.3, alpha=0.01, steps=40):
    attack = PGD(model, eps=eps, alpha=alpha, steps=steps)
    correct = 0
    total = 0

    for images, _ in data_loader:
        images = images.to(device)
        target_labels = torch.full((images.size(0),), target_class, dtype=torch.long).to(device)

        adv_images = attack(images, target_labels)
        outputs = model(adv_images)
        _, predicted = outputs.max(1)

        correct += (predicted == target_labels).sum().item()
        total += target_labels.size(0)

    success_rate = 100 * correct / total
    print(f"Targeted Attack Success Rate (Target Class {target_class}): {success_rate:.2f}%")
    return success_rate


# Untargeted Attack
def untargeted_attack(model, data_loader, eps=0.3, alpha=0.01, steps=40):
    attack = PGD(model, eps=eps, alpha=alpha, steps=steps)
    correct = 0
    total = 0

    for images, _ in data_loader:
        images = images.to(device)
        labels = torch.randint(0, 10, (images.size(0),), dtype=torch.long).to(device)  # Random dummy labels for now

        adv_images = attack(images, labels)
        outputs = model(adv_images)
        _, predicted = outputs.max(1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Accuracy after Untargeted Attack: {accuracy:.2f}%")
    return accuracy


# Perform Attacks
print("Running Targeted Attack...")
targeted_attack(model, kitti_loader, target_class=2)  # Example target class

print("\nRunning Untargeted Attack...")
untargeted_attack(model, kitti_loader)