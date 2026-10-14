import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

x = torch.randn(1000, 1000).to(device)

print(x.device)