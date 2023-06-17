import torch
import sys

print(torch.cuda.Device)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torch.load(sys.argv[1])
net = net.to(device)
print(net)
dummy_input = torch.Tensor(1,9,84,84).to(device, dtype=torch.float32)
print(dummy_input.type())
torch.onnx.export(net, dummy_input, './test_model.onnx',export_params=True,opset_version=11,do_constant_folding=True,input_names=['input'],output_names=['output'])
print(torch.onnx.producer_version)