import torch

x = torch.tensor([1, 2], dtype=torch.float32, requires_grad=True)
y = 2*(x + 1)
z = torch.mean(y ** 2)
z.backward(retain_graph=True, create_graph=True)
print(x.grad) # 输出[8, 12]
x.grad.data.zero_()
x.grad.backward(torch.ones(x.grad.size()))
print(x.grad) # 期望输出[4, 4], 实际输出[12, 16]，看起来是把一阶导二阶导加起来了……