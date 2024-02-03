import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.quantization import quantize, QuantStub, DeQuantStub, quantize_dynamic


# Define the dataset and dataloaders
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Assuming you have 'accident' and 'not_accident' as classes directly under your current working directory
train_dataset = torchvision.datasets.ImageFolder(root='.', transform=data_transform)
val_dataset = torchvision.datasets.ImageFolder(root='.', transform=data_transform)

# Assuming you have a 'train' and 'val' subdirectories under each class
# Split the dataset into training and validation sets (adjust the split ratio as needed)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define ResNet-18 model
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Linear(512, 2)  # Binary classification, so modify the last layer

    def forward(self, x):
        return self.resnet18(x)

# Instantiate the model
model = ResNet18()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Print model weights after training
# print("\nModel Weights after training:")
# for name, param in model.named_parameters():
#     print(f"{name}: {param}")

# Evaluate the model before quantization
model.eval()
correct_before_quantization = 0
total_before_quantization = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_before_quantization += labels.size(0)
        correct_before_quantization += (predicted == labels).sum().item()

accuracy_before_quantization = correct_before_quantization / total_before_quantization
print(f'Accuracy before quantization: {accuracy_before_quantization}')


# ...






# quantized_model = quantize_dynamic(model)
# model.qconfig = torch.quantization.get_default_qconfig('x86')
# model_fp32_prepared = torch.quantization.prepare(model)
# model_fp32_fused = torch.quantization.fuse_modules(model_fp32_prepared, [['resnet18.conv1', 'resnet18.bn1']])
# quantized_model = torch.quantization.convert(model_fp32_fused)
qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Prepare the model for quantization
model_fp32_prepared = torch.quantization.prepare(model, qconfig=qconfig)

# Convert the prepared model to a quantized model
quantized_model = torch.quantization.convert(model_fp32_prepared)




# print("\nModel Weights after quantization:")
# for name, param in quantized_model.named_parameters():
#     print(f"{name}: {param}")

# ...


# Evaluate the quantized model
quantized_model.eval()
correct_after_quantization = 0
total_after_quantization = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs=quantized_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_after_quantization += labels.size(0)
        correct_after_quantization += (predicted == labels).sum().item()


accuracy_after_quantization = correct_after_quantization / total_after_quantization
print(f'Accuracy after quantization: {accuracy_after_quantization}')

# Save the quantized model in a .pth file
torch.save(quantized_model.state_dict(), 'quantized_resnet18_model.pth')
