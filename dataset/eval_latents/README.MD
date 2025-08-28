
This tensor is created by the following code for reproducibility.

```python
import torch
generator = torch.Generator(device="cuda").manual_seed(0)
shape = (16, 64, 64) # h=512, w=512
latents = []
for i in range(32): # batchsize=32
    latents.append(torch.randn(shape, generator=generator, device="cuda").unsqueeze(0))

latents = torch.cat(latents, dim=0)
torch.save(latents, "dataset/eval_latents/tensor.pt")
```

