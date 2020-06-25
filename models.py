import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = np.finfo(np.float32).tiny


# binary concrete selector
def binary_concrete(x, t, hard=False, use_noise=True):
    if use_noise:
        u = torch.zeros_like(x.data).uniform_(0, 1) + torch.tensor([EPSILON]).to(x.device)
        logs = u.log() - (-u).log1p()
        z_soft = torch.sigmoid((x + logs) / t)
    else:
        z_soft = torch.sigmoid(x / t)

    # Straight through estimator
    if hard:
        z = (z_soft > 0.5).float() - z_soft.detach() + z_soft
    else:
        z = z_soft

    return z


# analytical approximation to the Heaviside with a logistic function
def heaviside_logistic(x, k):
    return torch.sigmoid(2. * k * x)


# approximate a unit height boxcar function
# using analytic approximations of the Heaviside function
def unitboxcar(x, mu, l, heaviside, k):
    # parameterize boxcar function by the center and length
    dist = x - mu
    window_half = l / 2.
    y = heaviside(dist + window_half, k) - heaviside(dist - window_half, k)
    return y


# approximate a unit height boxcar function
# using analytic approximations of the Heaviside function
def unitboxcar_center(dist, l, heaviside, k):
    # parameterize boxcar function by the center and length
    window_half = l / 2.
    y = heaviside(dist + window_half, k) - heaviside(dist - window_half, k)
    return y


class SpatialAgg(nn.Module):
    """
    Aggregate time-series based on phylogenetic distance.
    We use the heavyside logistic function to calculate the
    importance weights of each OTU for a detector and then normalize.
    """
    def __init__(self, num_rules, num_detectors, dist):
        super(SpatialAgg, self).__init__()
        # Initialize phylo. distance matrix as a parameter
        # with requires_grad = False
        self.dist = nn.Parameter(
            torch.from_numpy(dist),
            requires_grad=False)

        # OTU selection bandwidth
        # All OTUs within kappa radius are deemed to be selected
        self.kappa = nn.Parameter(torch.Tensor(num_rules, num_detectors))

    def forward(self, x, k=1):
        # Compute unnormalized OTU weights
        otu_wts_unnorm = heaviside_logistic(self.kappa.pow(2).unsqueeze(-1) - self.dist, k)

        # Normalize OTU wts
        otu_wts = (otu_wts_unnorm).div(otu_wts_unnorm.sum(dim=-1, keepdims=True).clamp(1))

        if torch.isnan(otu_wts).any():
            print(otu_wts_unnorm.sum(dim=-1))
            print(self.kappa)
            raise ValueError('Nan in spatial aggregation!')

        # Aggregation of time-series along OTU dimension
        # Essentially a convolution operation
        x = torch.einsum('kij,sjt->skit', otu_wts, x)

        return x, otu_wts

    def init_params(self, init_args):
        # Initialize kappa parameter
        self.kappa.data = torch.from_numpy(init_args['kappa_init']).pow(0.5).float()

        return


class TimeAgg(nn.Module):
    """
    Aggregate time-series along the time dimension. Select a contiguous
    time window that's important for prediction task.
    We use the heavyside logistic function to calculate the
    importance weights of each time point for a detector and then normalize.
    """
    def __init__(self, num_rules, num_detectors, num_time):
        super(TimeAgg, self).__init__()
        # Tensor of time points, starting from 0 to num_time - 1 (experiment duration)
        self.times = nn.Parameter(torch.arange(num_time, dtype=torch.float32),
            requires_grad=False)

        # Time window center parameter
        self.mu = nn.Parameter(torch.Tensor(num_rules, num_detectors))

        # Time window bandwidth parameter
        self.sigma = nn.Parameter(torch.Tensor(num_rules, num_detectors))

    def forward(self, x, mask=None, k=1.):
        # Compute unnormalized importance weights for each time point
        time_wts_unnorm = unitboxcar(self.times,
            self.mu.unsqueeze(-1),
            self.sigma.pow(2).unsqueeze(-1),
            heaviside_logistic, k)

        # Mask out time points with no samples
        if mask is not None:
            time_wts_unnorm = time_wts_unnorm.mul(
                mask.unsqueeze(1).unsqueeze(1))

        # Normalize importance time weights
        time_wts = (time_wts_unnorm).div(time_wts_unnorm.sum(dim=-1, keepdims=True).clamp(1))

        if torch.isnan(time_wts).any():
            print(time_wts_unnorm.sum(dim=-1))
            print(self.mu)
            print(self.sigma)
            raise ValueError('Nan in time aggregation!')

        # Aggregation over time dimension
        # Essentially a convolution over time
        x_abun = x.mul(time_wts).sum(dim=-1)

        return x_abun, time_wts

    def init_params(self, init_args):
        # Initialize mu parameter
        self.mu.data = torch.from_numpy(init_args['mu_init']).float()

        # Initialize sigma parameter
        self.sigma.data = torch.from_numpy(init_args['sigma_init']).pow(0.5).float()

        return


class Threshold(nn.Module):
    """
    Learn a threshold abundance for the rules.
    The output is a gated response of whether the aggregated
    abundance from spatial and time aggregation steps is above/below
    the learned threshold.
    """
    def __init__(self, num_rules, num_detectors):
        super(Threshold, self).__init__()
        # Parameter for learnable threshold abundance
        self.thresh = nn.Parameter(torch.Tensor(num_rules, num_detectors))
    
    def forward(self, x, t=1, hard=False, thresh_k=1., bc_k=1., use_noise=True):
        # Response of the detector for avg abundance
        x_above = heaviside_logistic(x - self.thresh.pow(2), thresh_k)

        if torch.isnan(x_above).any():
            print(x_above)
            print(self.thresh)
            raise ValueError('Nan in threshold aggregation!')

        return x_above

    def init_params(self, init_args):
        # Initialize the threshold parameter
        self.thresh.data = torch.from_numpy(init_args['thresh_init']).pow(0.5).float()

        return


class Rules(nn.Module):
    """
    Combine the reponses of detectors to compute the
    approximate logical AND operation as the rule responses.
    """
    def __init__(self, num_rules, num_detectors):
        super(Rules, self).__init__()
        # Binary concrete selector variable for detectors
        self.alpha = nn.Parameter(torch.Tensor(num_rules, num_detectors))
        nn.init.normal_(self.alpha, 0, 1)

    def forward(self, x, t=1., hard=False, k=1., use_noise=True):
        if self.training:
            # Binary concrete for active detector selection
            # During training only
            if use_noise:
                z = binary_concrete(self.alpha, t, hard=hard)
            else:
                z = binary_concrete(self.alpha, t, hard=hard, use_noise=False)
        else:
            # Heavyside logistic for active rule selection
            # During evaluation only
            z = binary_concrete(self.alpha, 1 / k, hard=hard, use_noise=False)

        # Approximate logical AND operation
        x = (1 - z.mul(1 - x)).prod(dim=-1)

        self.z = z

        if torch.isnan(x).any():
            print(x)
            print(z)
            print(self.alpha)
            raise ValueError('Nan in rules aggregation!')

        return x, z

    def init_params(self, init_args):
        return
    

class DenseLayer(nn.Module):
    """Linear classifier for computing the predicted outcome."""
    def __init__(self, in_feat, out_feat):
        super(DenseLayer, self).__init__()
        # Logistic regression coefficients
        self.weight = nn.Parameter(torch.Tensor(out_feat, in_feat))

        # Logistic regression bias
        self.bias = nn.Parameter(torch.Tensor(out_feat))

        # Parameter for selecting active rules
        self.beta = nn.Parameter(torch.Tensor(in_feat))

        # Initialize weight parameter
        nn.init.kaiming_uniform_(self.weight)

        # Initialize bias parameter
        nn.init.zeros_(self.bias)

        # Initialize beta parameter
        nn.init.normal_(self.beta, 0, 1)

    def forward(self, x, t=1., hard=False, k=1., use_noise=True):
        if self.training:
            # Binary concrete for active rule selection
            # During training only
            if use_noise:
                z = binary_concrete(self.beta, t, hard)
            else:
                z = binary_concrete(self.beta, t, hard, use_noise=False)
        else:
            # Heavyside logistic for active rule selection
            # During evaluation only
            z = binary_concrete(self.beta, 1 / k, hard, use_noise=False)

        self.z_r = z

        # Predict the outcome
        x = F.linear(x, self.weight * (z.unsqueeze(0)), self.bias)

        if torch.isnan(x).any():
            print(x)
            print(z)
            print(self.weight)
            print(self.bias)
            raise ValueError('Nan in dense aggregation!')

        return x, z

    def init_params(self, init_args):
        return


class MyModel(nn.Module):
    """docstring for MyModel"""
    def __init__(self, num_rules, num_detectors, num_time, dist):
        super(MyModel, self).__init__()
        self.spat_attn = SpatialAgg(num_rules, num_detectors, dist)
        self.time_attn = TimeAgg(num_rules, num_detectors, num_time)
        self.thresh_func = Threshold(num_rules, num_detectors)
        self.rules = Rules(num_rules, num_detectors)
        self.fc = DenseLayer(num_rules, 2)

    def forward(self, x, mask=None, t=1., hard=False,
        otu_k=1.,
        time_k=1.,
        thresh_k=1.,
        bc_k=1.,
        use_noise=True):
        x, spat_wts = self.spat_attn(x, k=otu_k)
        x, time_wts = self.time_attn(x, mask=mask, k=time_k)
        x = self.thresh_func(x, t=t, hard=hard,
            thresh_k=thresh_k, bc_k=bc_k,
            use_noise=use_noise)
        x, z = self.rules(x, t=t, hard=hard, k=bc_k, use_noise=use_noise)
        x, z_r = self.fc(x, t=t, hard=hard, k=bc_k, use_noise=use_noise)
        return x, [spat_wts, time_wts, z, z_r]

    def init_params(self, init_args):
        for m in self.children():
            m.init_params(init_args)

        return