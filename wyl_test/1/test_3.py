import torch

t1 = torch.tensor([[[1, 2],
                    [3, 4]]])

t2 = torch.tensor([[[5, 6],
                    [7, 8]],
                   [[9, 10],
                    [11, 12]]])

t3 = torch.tensor([[[13, 14]]])

t4 = torch.tensor([[[15],
                    [16]]])

print(t1.shape)
print(t2.shape)
print(t3.shape)
print(t4.shape)

result = torch.cat((t1, t4), dim=2)
print(result)
print(result.shape)