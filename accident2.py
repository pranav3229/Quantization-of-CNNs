import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int32Bias


# class QuantWeightActBiasLeNet(Module):
#     def __init__(self):
#         super(LowPrecisionLeNet, self).__init__()
#         self.quant_inp = qnn.QuantIdentity(bit_width=4, return_quant_tensor=True)
#         self.conv1 = qnn.QuantConv2d(3, 6, 5, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
#         self.relu1 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
#         self.conv2 = qnn.QuantConv2d(6, 16, 5, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
#         self.relu2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
#         self.fc1   = qnn.QuantLinear(16*5*5, 120, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
#         self.relu3 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
#         self.fc2   = qnn.QuantLinear(120, 84, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
#         self.relu4 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
#         self.fc3   = qnn.QuantLinear(84, 10, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

#     def forward(self, x):
#         out = self.quant_inp(x)
#         out = self.relu1(self.conv1(out))
#         out = F.max_pool2d(out, 2)
#         out = self.relu2(self.conv2(out))
#         out = F.max_pool2d(out, 2)
#         out = out.reshape(out.shape[0], -1)
#         out = self.relu3(self.fc1(out))
#         out = self.relu4(self.fc2(out))
#         out = self.fc3(out)
#         return out

# quant_weight_act_bias_lenet = QuantWeightActBiasLeNet()


class Block(Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=4, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        out = self.quant_inp(x)
        identity = out
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        out += identity
        out = self.relu(out)
        return out
    

class ResNet_18(Module):
    
    def __init__(self, image_channels, num_classes):
        
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.quant_inp = qnn.QuantIdentity(bit_width=4, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(image_channels, 64, kernel_size=7, stride=2, padding=3,bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.maxpool = qnn.QuantMaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = qnn.TruncAdaptiveAvgPool2d((1, 1),bias=True,weight_bit_width=4,bias_quant=Int32Bias)
        self.fc = qnn.QuantLinear(512, num_classes,bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        
    def __make_layer(self, in_channels, out_channels, stride):
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):
        out = self.quant_inp(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out 
    
    def identity_downsample(self, in_channels, out_channels):
        
        return nn.Sequential(
            qnn.QuantConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,bias=True, weight_bit_width=4, bias_quant=Int32Bias), 
            nn.BatchNorm2d(out_channels)
        )
    

model=ResNet_18(3,2)


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
correct_after_quantization = 0
total_after_quantization = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_after_quantization += labels.size(0)
        correct_after_quantization += (predicted == labels).sum().item()

accuracy_after_quantization = correct_after_quantization / total_after_quantization
print(f'Accuracy after quantization: {accuracy_after_quantization}')


from brevitas.export import export_onnx_qcdq
# Weight-activation-bias model
export_onnx_qcdq(model, torch.randn(1, 3, 32, 32), export_path='4b_weight_act_bias_lenet.onnx')