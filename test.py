import torch
import json

# Load the model
model = torch.load('ultranet.pt')

# Convert the model's state_dict to a dictionary of numpy arrays
# model_dict = model.state_dict()
# model_dict = {k: v.numpy() for k, v in model_dict.items()}

traced_script_module = torch.jit.trace(model, torch.randn(1,3,224,224))
torch.jit.save(traced_script_module, "model.onnx")
