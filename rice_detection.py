import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from google.colab import drive
import matplotlib.pyplot as plt
import numpy as np

# Mount Google Drive
drive.mount('/content/drive')

# Define Patch Embedding Module
class PatchEmbedding(nn.Module):
    def _init_(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self)._init_()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

# Define Transformer Model
class TransformerModel(nn.Module):
    def _init_(self, img_size, patch_size, in_channels, embed_dim, num_heads, num_blocks, num_classes):
        super(TransformerModel, self)._init_()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                nn.LayerNorm(embed_dim),
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                )
            ) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # Patch embedding
        for block in self.blocks:
            attn_out, _ = block[1](x, x, x)  # MultiheadAttention
            x = x + attn_out  # Add residual
            x = block[3](x) + x  # MLP + residual
        x = self.norm(x.mean(dim=1))  # Global pooling
        return self.head(x)

# Dataset Path and Transforms
dataset_path = "/content/drive/My Drive/Rice_disease_data"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset and DataLoader
train_dataset = datasets.ImageFolder(dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Model Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(
    img_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=128,
    num_heads=8,
    num_blocks=4,
    num_classes=len(train_dataset.classes)
).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

# Calculate Final Accuracy and Display Example Outputs
model.eval()
correct = 0
total = 0
class_names = train_dataset.classes

# Example Output Storage
examples = []

with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Store a few example outputs
        if len(examples) < 5:  # Collect up to 5 examples
            examples.extend(zip(images.cpu(), predicted.cpu(), labels.cpu()))

accuracy = correct / total * 100
print(f"Final Training Accuracy: {accuracy:.2f}%")

# Display Example Outputs
plt.figure(figsize=(12, 6))
for i, (image, pred, label) in enumerate(examples[:5]):
    image = image.permute(1, 2, 0).numpy()  # Convert to HWC for matplotlib
    plt.subplot(1, 5, i + 1)
    plt.imshow(image)
    plt.title(f"Pred: {class_names[pred]}, True: {class_names[label]}")
    plt.axis('off')
plt.show()
