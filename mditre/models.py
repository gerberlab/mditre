import numpy as np
from scipy.special import logit

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lowest possible float
EPSILON = np.finfo(np.float32).tiny

# binary concrete selector
def binary_concrete(x, k, hard=False, use_noise=True):
    if use_noise:
        u = torch.zeros_like(x.data).uniform_(0, 1) + torch.tensor([EPSILON]).to(x.device)
        logs = u.log() - (-u).log1p()
        z_soft = torch.sigmoid((x + logs) * k)
    else:
        z_soft = torch.sigmoid(x * k)

    # Straight through estimator
    if hard:
        z = (z_soft > 0.5).float() - z_soft.detach() + z_soft
    else:
        z = z_soft

    return z


# approximate a unit height boxcar function
# using analytic approximations of the Heaviside function
def unitboxcar(x, mu, l, k):
    # parameterize boxcar function by the center and length
    dist = x - mu
    window_half = l / 2.
    y = torch.sigmoid((dist + window_half) * k) - torch.sigmoid((dist - window_half) * k)
    return y


def transf_log(x, u, l):
    return (u - l) * torch.sigmoid(x) + l


def inv_transf_log(x, u, l):
    return logit((x - l) / (u - l))


class SpatialAgg(nn.Module):
    """
    Aggregate time-series based on phylogenetic distance.
    We use the heavyside logistic function to calculate the
    importance weights of each OTU for a detector and then normalize.
    """
    def __init__(self, num_rules, num_otus, dist):
        super(SpatialAgg, self).__init__()
        # Initialize phylo. distance matrix as a parameter
        # with requires_grad = False
        self.register_buffer('dist', torch.from_numpy(dist))

        # OTU selection bandwidth
        # All OTUs within kappa radius are deemed to be selected
        self.kappa = nn.Parameter(torch.Tensor(num_rules, num_otus))

    def forward(self, x, k=1):
        # Compute unnormalized OTU weights
        kappa = transf_log(self.kappa, 0, self.dist.max().item()).unsqueeze(-1)
        otu_wts_unnorm = torch.sigmoid((kappa - self.dist) * k)

        self.wts = otu_wts_unnorm

        if torch.isnan(otu_wts_unnorm).any():
            print(otu_wts_unnorm.sum(dim=-1))
            print(self.kappa)
            raise ValueError('Nan in spatial aggregation!')

        # Aggregation of time-series along OTU dimension
        # Essentially a convolution operation
        x = torch.einsum('kij,sjt->skit', otu_wts_unnorm, x)

        return x

    def init_params(self, init_args):
        # Initialize kappa parameter
        self.kappa.data = torch.from_numpy(inv_transf_log(init_args['kappa_init'], 0, self.dist.max().item())).float()

        return


class SpatialAggDynamic(nn.Module):
    """
    Aggregate time-series based on phylogenetic distance.
    We use the heavyside logistic function to calculate the
    importance weights of each OTU for a detector and then normalize.
    """
    def __init__(self, num_rules, num_otu_centers, dist, emb_dim, num_otus):
        super(SpatialAggDynamic, self).__init__()
        # Initialize phylo. distance matrix as a parameter
        # with requires_grad = False
        self.num_rules = num_rules
        self.num_otu_centers = num_otu_centers
        self.emb_dim = emb_dim
        self.register_buffer('dist', torch.from_numpy(dist))

        # OTU centers
        self.eta = nn.Parameter(torch.Tensor(num_rules, num_otu_centers, emb_dim))

        # OTU selection bandwidth
        # All OTUs within kappa radius are deemed to be selected
        self.kappa = nn.Parameter(torch.Tensor(num_rules, num_otu_centers))

    def forward(self, x, k=1):
        # Compute unnormalized OTU weights
        kappa = self.kappa.exp().unsqueeze(-1)
        dist = (self.eta.reshape(self.num_rules, self.num_otu_centers, 1, self.emb_dim) - self.dist).norm(2, dim=-1)
        otu_wts_unnorm = torch.sigmoid((kappa - dist) * k)

        # if not self.training:
        #     otu_wts_unnorm = (kappa > dist).float()

        self.wts = otu_wts_unnorm

        if torch.isnan(otu_wts_unnorm).any():
            print(otu_wts_unnorm.sum(dim=-1))
            print(self.kappa)
            raise ValueError('Nan in spatial aggregation!')

        # Aggregation of time-series along OTU dimension
        # Essentially a convolution operation
        x = torch.einsum('kij,sjt->skit', otu_wts_unnorm, x)

        self.kappas = kappa
        self.emb_dist = dist

        return x

    def init_params(self, init_args):
        # Initialize kappa parameter
        self.kappa.data = torch.from_numpy(init_args['kappa_init']).log().float()
        self.eta.data = torch.from_numpy(init_args['eta_init']).float()

        return


class TimeAgg(nn.Module):
    """
    Aggregate time-series along the time dimension. Select a contiguous
    time window that's important for prediction task.
    We use the heavyside logistic function to calculate the
    importance weights of each time point for a detector and then normalize.
    """
    def __init__(self, num_rules, num_otus, num_time, num_time_centers):
        super(TimeAgg, self).__init__()
        # Tensor of time points, starting from 0 to num_time - 1 (experiment duration)
        self.num_time = num_time
        self.register_buffer('times', torch.arange(num_time, dtype=torch.float32))

        # # Time window bandwidth parameter
        self.abun_a = nn.Parameter(torch.Tensor(num_rules, num_otus))
        self.slope_a = nn.Parameter(torch.Tensor(num_rules, num_otus))
        self.abun_b = nn.Parameter(torch.Tensor(num_rules, num_otus))
        self.slope_b = nn.Parameter(torch.Tensor(num_rules, num_otus))

    def forward(self, x, mask=None, k=1.):
        # Compute unnormalized importance weights for each time point
        abun_a = torch.sigmoid(self.abun_a).unsqueeze(-1)
        slope_a = torch.sigmoid(self.slope_a).unsqueeze(-1)
        abun_b = torch.sigmoid(self.abun_b).unsqueeze(-1)
        slope_b = torch.sigmoid(self.slope_b).unsqueeze(-1)
        sigma = self.num_time * abun_a
        sigma_slope = self.num_time * slope_a
        mu = (self.num_time * abun_a / 2.) + (1 -  abun_a) * self.num_time * abun_b
        mu_slope = (self.num_time * slope_a / 2.) + (1 -  slope_a) * self.num_time * slope_b
        time_wts_unnorm = unitboxcar(self.times,
            mu,
            sigma, k)
        time_wts_unnorm_slope = unitboxcar(self.times,
            mu_slope,
            sigma_slope, k)

        # Mask out time points with no samples
        if mask is not None:
            time_wts_unnorm = time_wts_unnorm.mul(
                mask.unsqueeze(1).unsqueeze(1))
            time_wts_unnorm_slope = time_wts_unnorm_slope.mul(
                mask.unsqueeze(1).unsqueeze(1))

        # if not self.training:
        #     time_wts_unnorm = (time_wts_unnorm > 0.9).float()
        #     time_wts_unnorm_slope = (time_wts_unnorm_slope > 0.9).float()

        self.wts = time_wts_unnorm
        self.wts_slope = time_wts_unnorm_slope

        # Normalize importance time weights
        time_wts = (time_wts_unnorm).div(time_wts_unnorm.sum(dim=-1, keepdims=True) + 1e-8)

        if torch.isnan(time_wts).any():
            print(time_wts_unnorm.sum(-1))
            raise ValueError('Nan in time aggregation!')

        # Aggregation over time dimension
        # Essentially a convolution over time
        x_abun = x.mul(time_wts).sum(dim=-1)

        # Compute approx. avg. slope over time window
        tau = self.times - mu_slope
        a = (time_wts_unnorm_slope * x).sum(dim=-1)
        b = (time_wts_unnorm_slope * tau).sum(dim=-1)
        c = (time_wts_unnorm_slope).sum(dim=-1)
        d = (time_wts_unnorm_slope * x * tau).sum(dim=-1)
        e = (time_wts_unnorm_slope * (tau ** 2)).sum(dim=-1)
        num = ((a*b) - (c*d))
        den = ((b**2) - (e*c)) + 1e-8
        x_slope = num / den

        if torch.isnan(x_slope).any():
            print(time_wts_unnorm_slope.sum(dim=-1))
            print(x_slope)
            raise ValueError('Nan in time aggregation!')


        self.m = mu
        self.m_slope = mu_slope
        self.s_abun = sigma
        self.s_slope = sigma_slope

        return x_abun, x_slope

    def init_params(self, init_args):
        # # Initialize mu and sigma parameter
        self.abun_a.data = torch.from_numpy(logit(init_args['abun_a_init'])).float()
        self.abun_b.data = torch.from_numpy(logit(init_args['abun_b_init'])).float()
        self.slope_a.data = torch.from_numpy(logit(init_args['slope_a_init'])).float()
        self.slope_b.data = torch.from_numpy(logit(init_args['slope_b_init'])).float()

        return


class TimeAggAbun(nn.Module):
    """
    Aggregate time-series along the time dimension. Select a contiguous
    time window that's important for prediction task.
    We use the heavyside logistic function to calculate the
    importance weights of each time point for a detector and then normalize.
    """
    def __init__(self, num_rules, num_otus, num_time, num_time_centers):
        super(TimeAggAbun, self).__init__()
        # Tensor of time points, starting from 0 to num_time - 1 (experiment duration)
        self.num_time = num_time
        self.register_buffer('times', torch.arange(num_time, dtype=torch.float32))

        # # Time window bandwidth parameter
        self.abun_a = nn.Parameter(torch.Tensor(num_rules, num_otus))
        self.abun_b = nn.Parameter(torch.Tensor(num_rules, num_otus))

    def forward(self, x, mask=None, k=1.):
        # Compute unnormalized importance weights for each time point
        abun_a = torch.sigmoid(self.abun_a).unsqueeze(-1)
        abun_b = torch.sigmoid(self.abun_b).unsqueeze(-1)
        sigma = self.num_time * abun_a
        mu = (self.num_time * abun_a / 2.) + (1 -  abun_a) * self.num_time * abun_b
        time_wts_unnorm = unitboxcar(self.times,
            mu,
            sigma, k)

        # Mask out time points with no samples
        if mask is not None:
            time_wts_unnorm = time_wts_unnorm.mul(
                mask.unsqueeze(1).unsqueeze(1))

        # if not self.training:
        #     time_wts_unnorm = (time_wts_unnorm > 0.9).float()
        #     time_wts_unnorm_slope = (time_wts_unnorm_slope > 0.9).float()

        self.wts = time_wts_unnorm

        # Normalize importance time weights
        time_wts = (time_wts_unnorm).div(time_wts_unnorm.sum(dim=-1, keepdims=True) + 1e-8)

        if torch.isnan(time_wts).any():
            print(time_wts_unnorm.sum(-1))
            raise ValueError('Nan in time aggregation!')

        # Aggregation over time dimension
        # Essentially a convolution over time
        x_abun = x.mul(time_wts).sum(dim=-1)

        self.m = mu
        self.s_abun = sigma

        return x_abun

    def init_params(self, init_args):
        # # Initialize mu and sigma parameter
        self.abun_a.data = torch.from_numpy(logit(init_args['abun_a_init'])).float()
        self.abun_b.data = torch.from_numpy(logit(init_args['abun_b_init'])).float()

        return


class Threshold(nn.Module):
    """
    Learn a threshold abundance for each detector.
    The output is a sharp but smooth gated response of
    whether the aggregated abundance from previous steps
    is above/below the learned threshold.
    """
    def __init__(self, num_rules, num_otus, num_time_centers):
        super(Threshold, self).__init__()
        # Parameter for learnable threshold abundance
        self.thresh = nn.Parameter(torch.Tensor(num_rules, num_otus))
    
    def forward(self, x, k=1):
        # Response of the detector for avg abundance
        # if not self.training:
        #     x = (x > self.thresh).float()
        # else:
        x = torch.sigmoid((x - self.thresh) * k)

        return x

    def init_params(self, init_args):
        # Initialize the threshold parameter
        self.thresh.data = torch.from_numpy(init_args['thresh_init']).float()

        return


class Slope(nn.Module):
    """
    Learn a threshold abundance for the rules.
    The output is a gated response of whether the aggregated
    abundance from spatial and time aggregation steps is above/below
    the learned threshold.
    """
    def __init__(self, num_rules, num_otus, num_time_centers):
        super(Slope, self).__init__()
        # Parameter for learnable threshold abundance
        self.slope = nn.Parameter(torch.Tensor(num_rules, num_otus))
    
    def forward(self, x, k=1):
        # Response of the detector for avg abundance
        # if not self.training:
        #     x = (x > self.slope).float()
        # else:
        x = torch.sigmoid((x - self.slope) * k)

        if torch.isnan(self.slope).any():
            print(self.slope)
            raise ValueError('Nan in slope!')

        return x

    def init_params(self, init_args):
        # Initialize the threshold parameter
        self.slope.data = torch.from_numpy(init_args['slope_init']).float()

        return


class Rules(nn.Module):
    """
    Combine the reponses of detectors to compute the
    approximate logical AND operation as the rule responses.
    """
    def __init__(self, num_rules, num_otus, num_time_centers):
        super(Rules, self).__init__()
        # Binary concrete selector variable for detectors
        self.alpha = nn.Parameter(torch.Tensor(num_rules, num_otus))

    def forward(self, x, k=1., hard=False, use_noise=True):
        if self.training:
            # Binary concrete for detector selection
            if use_noise:
                z = binary_concrete(self.alpha, k, hard=hard)
            else:
                z = binary_concrete(self.alpha, k, hard=hard, use_noise=False)
        else:
            z = binary_concrete(self.alpha, k, hard=hard, use_noise=False)
            # z = (z > 0.9).float()

        self.x = x

        # Approximate logical AND operation
        x = (1 - z.mul(1 - x)).prod(dim=-1)

        self.z = z

        return x

    def init_params(self, init_args):
        self.alpha.data = torch.from_numpy(init_args['alpha_init']).float()
    

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


    def forward(self, x, x_slope, k=1., hard=False, use_noise=True):
        if self.training:
            # Binary concrete for rule selection
            if use_noise:
                z = binary_concrete(self.beta, k, hard=hard)
            else:
                z = binary_concrete(self.beta, k, hard=hard, use_noise=False)
        else:
            # Heavyside logistic for active rule selection
            # During evaluation only
            z = binary_concrete(self.beta, k, hard=hard, use_noise=False)
            # z = (z > 0.9).float()


        self.sub_log_odds = ((x * x_slope) * ((self.weight * z.unsqueeze(0)).reshape(-1))) + self.bias

        # Predict the outcome
        x = F.linear(x * x_slope, self.weight * z.unsqueeze(0), self.bias)

        self.z = z

        self.log_odds = x.squeeze(-1)

        return x.squeeze(-1)

    def init_params(self, init_args):
        self.weight.data = torch.from_numpy(init_args['w_init']).float()
        self.bias.data = torch.from_numpy(init_args['bias_init']).float()
        self.beta.data = torch.from_numpy(init_args['beta_init']).float()


class DenseLayerAbun(nn.Module):
    """Linear classifier for computing the predicted outcome."""
    def __init__(self, in_feat, out_feat):
        super(DenseLayerAbun, self).__init__()
        # Logistic regression coefficients
        self.weight = nn.Parameter(torch.Tensor(out_feat, in_feat))

        # Logistic regression bias
        self.bias = nn.Parameter(torch.Tensor(out_feat))

        # Parameter for selecting active rules
        self.beta = nn.Parameter(torch.Tensor(in_feat))


    def forward(self, x, k=1., hard=False, use_noise=True):
        if self.training:
            # Binary concrete for rule selection
            if use_noise:
                z = binary_concrete(self.beta, k, hard=hard)
            else:
                z = binary_concrete(self.beta, k, hard=hard, use_noise=False)
        else:
            # Heavyside logistic for active rule selection
            # During evaluation only
            z = binary_concrete(self.beta, k, hard=hard, use_noise=False)
            # z = (z > 0.9).float()


        self.sub_log_odds = ((x) * ((self.weight * z.unsqueeze(0)).reshape(-1))) + self.bias

        # Predict the outcome
        x = F.linear(x, self.weight * z.unsqueeze(0), self.bias)

        self.log_odds = x.squeeze(-1)

        self.z = z

        return x.squeeze(-1)

    def init_params(self, init_args):
        self.weight.data = torch.from_numpy(init_args['w_init']).float()
        self.bias.data = torch.from_numpy(init_args['bias_init']).float()
        self.beta.data = torch.from_numpy(init_args['beta_init']).float()


class MDITRE(nn.Module):
    """docstring for MDITRE"""
    def __init__(self, num_rules, num_otus, num_otu_centers,
            num_time, num_time_centers, dist, emb_dim):
        super(MDITRE, self).__init__()
        self.spat_attn = SpatialAggDynamic(num_rules, num_otu_centers, dist, emb_dim, num_otus)
        self.time_attn = TimeAgg(num_rules, num_otu_centers, num_time, num_time_centers)
        self.thresh_func = Threshold(num_rules, num_otu_centers, num_time_centers)
        self.slope_func = Slope(num_rules, num_otu_centers, num_time_centers)
        self.rules = Rules(num_rules, num_otu_centers, num_time_centers)
        self.rules_slope = Rules(num_rules, num_otu_centers, num_time_centers)
        self.fc = DenseLayer(num_rules, 1)

    def forward(self, x, mask=None, k_alpha=1, k_beta=1,
        k_otu=1., k_time=1., k_thresh=1., k_slope=1.,
    	hard=False, use_noise=True):
        x = self.spat_attn(x, k=k_otu)
        x, x_slope = self.time_attn(x, mask=mask, k=k_time)
        x = self.thresh_func(x, k=k_thresh,)
        x = self.rules(x, hard=hard, k=k_alpha, use_noise=use_noise)
        x_slope = self.slope_func(x_slope, k=k_slope,)
        x_slope = self.rules_slope(x_slope, hard=hard, k=k_alpha, use_noise=use_noise)
        x = self.fc(x, x_slope, hard=hard, k=k_beta, use_noise=use_noise)
        return x

    def init_params(self, init_args):
        for m in self.children():
            m.init_params(init_args)

        return


class MDITREAbun(nn.Module):
    """docstring for MDITRE"""
    def __init__(self, num_rules, num_otus, num_otu_centers,
            num_time, num_time_centers, dist, emb_dim):
        super(MDITREAbun, self).__init__()
        self.spat_attn = SpatialAggDynamic(num_rules, num_otu_centers, dist, emb_dim, num_otus)
        self.time_attn = TimeAggAbun(num_rules, num_otu_centers, num_time, num_time_centers)
        self.thresh_func = Threshold(num_rules, num_otu_centers, num_time_centers)
        self.rules = Rules(num_rules, num_otu_centers, num_time_centers)
        self.fc = DenseLayerAbun(num_rules, 1)

    def forward(self, x, mask=None, k_alpha=1, k_beta=1,
        k_otu=1., k_time=1., k_thresh=1., k_slope=1.,
        hard=False, use_noise=True):
        x = self.spat_attn(x, k=k_otu)
        x = self.time_attn(x, mask=mask, k=k_time)
        x = self.thresh_func(x, k=k_thresh,)
        x = self.rules(x, hard=hard, k=k_alpha, use_noise=use_noise)
        x = self.fc(x, hard=hard, k=k_beta, use_noise=use_noise)
        return x

    def init_params(self, init_args):
        for m in self.children():
            m.init_params(init_args)

        return