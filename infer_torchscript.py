
import torch

model = torch.jit.load('model.pt', map_location=torch.device('cpu'))
wave_input = torch.randn(1, 80000)
text_input = torch.randint(high=1000, size=(1, 10))
outputs = model(wave_input, text_input)

print(outputs)

