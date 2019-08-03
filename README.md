# AutomaticLossWeightingPyTorch
PyTorch implementation of
[Multi-Task Learning Using Uncertainty to Weigh Losses
for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115 "arXiv abstract link") weighted loss module.

# Who Should Use This?
Anyone who has a multiple output prediction problem and wants to balance losses automatically. 

# Example Usage

```
from weighted_loss import WeightedLoss


def train(*args, **kwargs):
    device =  torch.device("cuda:0" if torch.cuda.
                                   is_available() else "cpu")
    reg_criterion1 = nn.SmoothL1Loss()
    reg_criterion2 = nn.SmoothL1Loss()
    cls_criterion1 = nn.BCEWithLogitsLoss() # Can be arbitraty number of losses
    weighed_criterion = WeightedLoss(n_reg_losses=3, n_cls_losses=1).to(device)
    # don't forget to pass loss parameters to the optimizer!
    params_to_optimize = [
        {
            'params': model.parameters(),
            'lr': 5e-4
        }, {
            'params': weighed_criterion.parameters(),
            'lr': 1e-1 # High learning rate for quick adaptation
        }
    ]
    optimizer = optim.Adam(params_to_optimize) # Your choice of optimizer
    ...
    any other setup you might want to do
    ...
    # Inside training loop:
        reg_loss1 = reg_criterion1(output1, label1)
        reg_loss2 = reg_criterion2(output2, label2)
        cls_loss1 = cls_criterion1(output3, label3)
        reg_losses = [reg_loss1, reg_loss2]
        cls_losses = [cls_loss1]
        loss = weighed_criterion(reg_losses, cls_losses)
        loss.backward()
        optimizer.step()
```
