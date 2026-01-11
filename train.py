import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Download the dataset
print("📁 Downloading Microsoft Cats vs Dogs dataset ...")

dataset = load_dataset("microsoft/cats_vs_dogs")
train_data = dataset['train']

print(f"✅ Dataset loaded!")
print(f"📊 Total images: {len(train_data)}")
print(f"🏷️  Labels: 0 = Cat, 1 = Dog\n")

# Step 2: Prepare the images
print("🖼️  Preparing image transformations...")

transform = transforms.Compose([
    transforms.Resize((128, 128)),      # Resize to 128x128
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize(               # Normalize colors
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Create a PyTorch Dataset wrapper
class CatDogDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['labels']
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Split into train and test 
train_size = int(0.8 * len(train_data))
test_size = len(train_data) - train_size
train_subset = train_data.select(range(train_size))
test_subset = train_data.select(range(train_size, len(train_data)))
train_dataset = CatDogDataset(train_subset, transform=transform)
test_dataset = CatDogDataset(test_subset, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"✅ Training images: {len(train_dataset)}")
print(f"✅ Test images: {len(test_dataset)}\n")

# Step 3: Build a simple CNN
class DogCatClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolutional layers 
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 colors -> 32 features
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce size: 128x128 -> 64x64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32 -> 64 features
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64 -> 128 features
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Reduce overfitting
            nn.Linear(256, 2)  # 2 classes: cat or dog
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Step 4: Create the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"💻 Using: {device.upper()}\n")

model = DogCatClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train one epoch
def train_one_epoch():
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        correct += (predictions.argmax(1) == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
    return 100 * correct / total, total_loss / len(train_loader)

# Step 6: Test the model
def test_model():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            correct += (predictions.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    return 100 * correct / total

# Step 7: Train the model
print("🚀 Training started...\n")
epochs = 5

for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}/{epochs}")
    train_acc, train_loss = train_one_epoch()
    test_acc = test_model()
    print(f"  ✅ Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%\n")

print(f"✨ Training complete! Final test accuracy: {test_acc:.2f}%\n")

# Step 8: Show predictions
def show_predictions():
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        predictions = model(images).argmax(1)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    class_names = ['Cat', 'Dog']
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].cpu() * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = img.clip(0, 1)
            
            true_label = class_names[labels[i].item()]
            pred_label = class_names[predictions[i].item()]
            
            ax.imshow(img)
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'True: {true_label}\nPredicted: {pred_label}', 
                        color=color, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dog_cat_predictions.png', dpi=150, bbox_inches='tight')
    print("📊 Predictions saved to 'dog_cat_predictions.png'")

show_predictions()

# Step 9: Save the model
torch.save(model.state_dict(), 'dog_cat_classifier.pth')
print("💾 Model saved to 'dog_cat_classifier.pth'")