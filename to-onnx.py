import torch
import torchvision.transforms as transforms
from torch.quantization import quantize, QuantStub, DeQuantStub

# Define ResNet-18 model
class ResNet18(torch.nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        self.resnet18.fc = torch.nn.Linear(512, 2)  # Binary classification, so modify the last layer

    def forward(self, x):
        return self.resnet18(x)

# Instantiate the model
model = ResNet18()

# Add quantization stubs
model.quant = QuantStub()
model.dequant = DeQuantStub()

# Load the saved quantized model state_dict
quantized_model_state_dict = torch.load('quantized_resnet18_model.pth')
model.load_state_dict(quantized_model_state_dict)

# Set the model to evaluation mode
model.eval()

# Create a dummy input with the appropriate shape
dummy_input = torch.randn(1, 3, 224, 224)

# Export the quantized model to ONNX format
onnx_path = 'quantized_resnet18_model.onnx'
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)
