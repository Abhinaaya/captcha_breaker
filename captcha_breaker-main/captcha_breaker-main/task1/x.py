import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = './data'
image_size = 128
data_transform = transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)

train_size = int(0.8* len(full_dataset))
val_size = (len(full_dataset) - train_size)//2
test_size = len(full_dataset) - train_size -  val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 100)
        )
    
    def forward(self,x):
        x =self.features(x)
        x = self.dense_layers(x)
        return x
    
model = Classifier().to(device)

loss_func = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.0003)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = loss_func(outputs, labels)
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        running_loss += loss.item()
        max_val, prediction = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (prediction==labels).sum().item()
        
    train_loss = running_loss/len(train_loader)
    train_accuracy = (correct/total)*100
    
    
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = loss_func(outputs, labels)
        
        val_loss += loss.item()
        max_val, prediction = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()
        
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
        
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
    
    
model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = loss_func(outputs, labels)
        
        test_loss += loss.item()
        max_val, prediction = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()
        
test_loss = test_loss / len(test_loader)
test_acc = 100 * correct / total

print(f'\nTest Results:')
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')