
This tensor is created by the following code for reproducibility.

```python
import torch
generator = torch.Generator().manual_seed(0)

# h=512, w=512
# T+1=41, B=32, d=16*64*64
shape = (41, 32, 16, 64, 64) 
latents = torch.randn(shape, generator=generator)
torch.save(latents, "dataset/eval_noise/tensor.pt")
```

