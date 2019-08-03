import torch
import torch.nn as nn
import random


class WeightedLoss(nn.Module):
    """
    Class that implements automatically weighed loss from:
    https://arxiv.org/pdf/1705.07115.pdf
    NOTE:
    Don't forget to give these params to the optimiser:
    optim.SGD(model.parameters() + criterion.parameters(), optim_args).
    """

    def __init__(self, n_reg_losses=0, n_cls_losses=0):

        super(WeightedLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.
                                   is_available() else "cpu")
        self.reg_coeffs = []
        self.cls_coeffs = []
        for i in range(n_reg_losses):
            init_value = random.random()  # Any init value will do.
            param = nn.Parameter(torch.tensor(init_value))
            name = "reg_param_" + str(i)
            self.register_parameter(name, param)
            self.reg_coeffs.append(param)
        for i in range(n_cls_losses):
            init_value = random.random()
            param = nn.Parameter(torch.tensor(init_value))
            name = "cls_param_" + str(i)
            self.register_parameter(name, param)
            self.cls_coeffs.append(param)

    def forward(self, reg_losses=[], cls_losses=[]):
        '''Forward pass

        Keyword Arguments:
            reg_losses {list} -- List of tensors of regression
            (Tested with smooth L1 and L2) losses (default: {[]})
            cls_losses {list} -- List of tensors of classification
            (tested with BCE) losses (default: {[]})

        Returns:
            torch.Tensor -- 0-dimensional tensor with final loss.
        '''

        assert len(reg_losses) == len(
            self.
            reg_coeffs), "Loss mismatch, check how many reg_losses are passed"
        assert len(cls_losses) == len(
            self.
            cls_coeffs), "Loss mismatch, check how many cls_losses are passed"
        net_loss = torch.zeros(1).to(self.device)
        for i, reg_loss in enumerate(reg_losses):
            net_loss += 0.5 * torch.exp(-self.reg_coeffs[i]) * reg_loss
            net_loss += 0.5 * self.reg_coeffs[i]
        for i, cls_loss in enumerate(cls_losses):
            net_loss += torch.exp(-self.cls_coeffs[i]) * cls_loss
            net_loss += 0.5 * self.cls_coeffs[i]
        return net_loss
