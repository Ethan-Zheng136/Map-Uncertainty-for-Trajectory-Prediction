import torch
from torch import nn as nn
from torch.nn import functional as F
from mmdet.models.losses import l1_loss, smooth_l1_loss
from mmdet.models.losses.utils import weighted_loss
import mmcv

from mmdet.models.builder import LOSSES
from torch.distributions.laplace import Laplace

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def pts_nll_loss_laplace(pred, target):
    # sample_size = target.shape[0] * target.shape[1]  # number of 2-D points

    # Reshape predictions and targets
    pts_reshape = pred[:, 0:40].reshape(-1, 2)     
    betas_reshape = pred[:, 40:80].reshape(-1, 2)
    target = target.reshape(-1, 2)

    m = Laplace(pts_reshape, betas_reshape)

    nll = -m.log_prob(target)
    # breakpoint()

    return nll 

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def pts_nll_loss_multivariategaussian(pred, target):
    target = target.reshape(-1, 2)
    sample_size = target.shape[0]  # number of 2-D points

    pts_pred = pred[:, :40].reshape(-1, 2)                   # [x_mean, y_mean]
    pts_betas = pred[:, 40:80].reshape(-1, 2)
    corr = pred[:, 80:100].reshape(-1, 1).squeeze()
    corr = torch.clamp(corr, -1 + 1e-5, 1 - 1e-5)  # [corr]

    var_x = pts_betas[:, 0] ** 2 + 1e-6  
    var_y = pts_betas[:, 1] ** 2 + 1e-6 
    cov_xy = corr * pts_betas[:, 0] * pts_betas[:, 1]       
    cov_matrix = torch.zeros(sample_size, 2, 2, device=pred.device)
    cov_matrix[:, 0, 0] = var_x
    cov_matrix[:, 1, 1] = var_y
    cov_matrix[:, 0, 1] = cov_xy
    cov_matrix[:, 1, 0] = cov_xy

    m = MultivariateNormal(pts_pred, covariance_matrix=cov_matrix)

    reg_loss = torch.mean(torch.abs(pts_betas))
    nll = -m.log_prob(target).unsqueeze(-1)
    # print('nll:', nll.shape)
    nll = nll + 0.01 * reg_loss
    
    return nll


@LOSSES.register_module()
class PtsNLLLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsNLLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        # breakpoint()
        weight = weight.view(-1, 2)   # 400, 40 -> 8000,2
        # loss_bbox = self.loss_weight * pts_nll_loss_laplace(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        loss_bbox = pts_nll_loss_multivariategaussian(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        # pred = 400, 80; target= 400, 40; 
        # breakpoint()
        
        num_points = pred.shape[-1] // 4
        loss = loss_bbox / num_points

        return loss*self.loss_weight


@LOSSES.register_module()
class LinesL1Loss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, beta=0.5):
        """
            L1 loss. The same as the smooth L1 loss
            Args:
                reduction (str, optional): The method to reduce the loss.
                    Options are "none", "mean" and "sum".
                loss_weight (float, optional): The weight of loss.
        """

        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
                shape: [bs, ...]
            target (torch.Tensor): The learning target of the prediction.
                shape: [bs, ...]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None. 
                it's useful when the predictions are not all valid.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        if self.beta > 0:
            loss = smooth_l1_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)
        
        else:
            loss = l1_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        # pred.shape = 400, 40, target.shape = 400. 40, weight.shape = 400, 40, reduction='mean', avg_factor=19.0
        # breakpoint()
        
        num_points = pred.shape[-1] // 2
        loss = loss / num_points

        return loss*self.loss_weight     # 9.336 * 50


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def bce(pred, label, class_weight=None):
    """
        pred: B,nquery,npts
        label: B,nquery,npts
    """

    if label.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == label.size()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')

    return loss


@LOSSES.register_module()
class MasksLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MasksLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            xxx
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = bce(pred, target, weight, reduction=reduction,
                   avg_factor=avg_factor)

        return loss*self.loss_weight

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def ce(pred, label, class_weight=None):
    """
        pred: B*nquery,npts
        label: B*nquery,
    """

    if label.numel() == 0:
        return pred.sum() * 0

    loss = F.cross_entropy(
        pred, label, weight=class_weight, reduction='none')

    return loss


@LOSSES.register_module()
class LenLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(LenLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            xxx
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = ce(pred, target, weight, reduction=reduction,
                   avg_factor=avg_factor)

        return loss*self.loss_weight