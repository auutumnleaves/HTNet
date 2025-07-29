import torch

a = torch.randn((2,3,4))

print(a)
print("===========")
print(torch.mean(a))
print("===========")
print(torch.mean(a, dim=0, keepdim=True))
print("----")
print(torch.mean(a, dim=0))
print("===========")
print(torch.mean(a, dim=1, keepdim=True))
print("----")
print(torch.mean(a, dim=1))
print("==========")
print(torch.mean(a, dim=2, keepdim=True))
print("----")
print(torch.mean(a, dim=2))