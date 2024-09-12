import sys
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.nn import init
from torch.autograd import Variable
import math
from copy import deepcopy
import multiprocessing

from mllp.utils import UnionFind

THRESHOLD_W = 0.5
THRESHOLD_Z = 0.5


"""Adaptation to L0 Reg

- Add constants limit_a, limit_b, epsilon
"""

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class RandomlyBinarize(torch.autograd.Function):
    """Implement the forward and backward propagation of the random binarization operation."""

    @staticmethod
    def forward(ctx, W, M):
        W = W.clone()
        W[M] = torch.where(W[M] > THRESHOLD_W, torch.ones_like(W[M]), torch.zeros_like(W[M]))
        ctx.save_for_backward(M.type(torch.float))
        return W

    @staticmethod
    def backward(ctx, grad_output):
        M, = ctx.saved_tensors
        grad_input = grad_output * (1.0 - M)
        return grad_input, None


class RandomBinarizationLayer(nn.Module):
    """Implement the Random Binarization (RB) method."""

    def __init__(self, shape, probability):
        super(RandomBinarizationLayer, self).__init__()
        self.shape = shape
        self.probability = probability
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.M = torch.rand(self.shape, device=self.device) < self.probability

    def forward(self, W):
        return RandomlyBinarize.apply(W, self.M)

    def refresh(self):
        self.M = torch.rand(self.shape, device=self.device) < self.probability

class L0ConjunctionLayer(nn.Module):
    """The conjunction layer is used to learn the rules."""

    def __init__(self, in_features, out_features, random_binarization_rate, use_not=False, bias=False, weight_decay=1., droprate_init=0.5, temperature=2./3., lamba=0.1, group_l0= False, **kwargs): # UPDATED
        super(L0ConjunctionLayer, self).__init__()
        self.use_not = use_not
        self.node_activation_cnt = None

        self.in_features = in_features if not use_not else in_features * 2 # UPDATED
        self.out_features = out_features # UPDATED
        self.weights = Parameter(0.1 * torch.rand(in_features, out_features)) # UPDATED
        self.randomly_binarize_layer = RandomBinarizationLayer(self.weights.shape, random_binarization_rate) # UPDATED
        self.group_l0 = group_l0 # NEW
        if self.group_l0:
            self.qz_loga = Parameter(torch.Tensor(in_features)) # NEW
        else:
            self.qz_loga = Parameter(torch.Tensor(in_features, out_features)) # NEW
        self.prior_prec = weight_decay # NEW
        self.temperature = temperature # NEW
        # self.droprate_init = droprate_init if droprate_init != 0. else 0.5 # NEW
        self.droprate_init = droprate_init # NEW
        self.lamba = lamba # NEW
        self.use_bias = False # NEW
        if bias: # NEW
            self.bias = Parameter(torch.Tensor(out_features)) # NEW
            self.use_bias = True # NEW
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor # NEW
        self.reset_parameters() # NEW

    def forward(self, input, randomly_binarize=False): # UPDATED
        if self.use_not:
            input= torch.cat((input, 1 - input), dim=1)
        # if self.local_rep or not self.training: # NEW
        if not self.training: # NEW
            # z = self.sample_z(input.size(0), sample=self.training) # NEW
            z = self.sample_z(1, sample=self.training) # NEW
            # xin = input.mul(z) # NEW
            # processed_input = xin # NEW
            processed_input = input # NEW
            # weights = self.weights # NEW
            weights = z.view(self.in_features, 1) * self.weights # NEW
            # output = xin.mm(self.weights) # NEW
        else:
            weights = self.sample_weights() # NEW
            processed_input = input # NEW
            # output = input.mm(weights) # NEW
        weights = self.randomly_binarize_layer(weights) if randomly_binarize else weights # UPDATED
        output = torch.prod((1 - (1 - processed_input)[:, :, None] * weights[None, :, :]), dim=1) # NEW
        # if self.use_bias: # NEW
        #     output.add_(self.bias) # NEW
        return output

    def binarized_forward(self, x): # UPDATED
        with torch.no_grad():
            if self.use_not:
                x = torch.cat((x, 1 - x), dim=1)
            x = x.type(torch.int)
            z = self.sample_z(1, sample=False) # NEW
            zb = torch.where(z > THRESHOLD_Z, torch.ones_like(z), torch.zeros_like(z)).type(torch.int) # NEW
            # xin = x.mul(zb) # NEW
            # processed_input = xin # NEW
            processed_input = x
            Wb = zb.view(self.in_features, 1) * torch.where(self.weights > THRESHOLD_W, torch.ones_like(self.weights), torch.zeros_like(self.weights)).type(torch.int) # UPDATED
            # Wb = torch.where(self.qz_loga > 0.0, torch.ones_like(self.qz_loga), torch.zeros_like(self.qz_loga)).type(torch.int) # UPDATED
            return torch.prod((1 - (1 - processed_input)[:, :, None] * Wb[None, :, :]), dim=1) # UPDATED

    def reset_parameters(self): # NEW
        # init.kaiming_normal(self.weights, mode='fan_out')

        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs): # NEW
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x): # NEW
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def one_minus_cdf_qz0(self):
        # let's use what is described in the paper (it's more specific, while the above is more general)
        return F.sigmoid(self.qz_loga - self.temperature * math.log(-limit_a / limit_b))

    def quantile_concrete(self, x): # NEW
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self): # NEW
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        if self.group_l0:
            logpw_col = torch.sum(- (.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 1)
            logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
            logpb = 0 if not self.use_bias else - torch.sum(.5 * self.prior_prec * self.bias.pow(2))
            return logpw + logpb
        else:
            lc = torch.sum(self.one_minus_cdf_qz0())
            return lc

    def regularization(self): # NEW
        if self.group_l0:
            return self._reg_w()
        else:
            return self._reg_w() * self.lamba

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.out_features
        expected_l0 = ppos * self.out_features
        if self.use_bias:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        return expected_flops.item(), expected_l0.item()

    def get_eps(self, size): # NEW
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True): # NEW
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            if self.group_l0:
                eps = self.get_eps(self.floatTensor(batch_size, self.in_features))
            else:
                eps = self.get_eps(self.floatTensor(batch_size, self.in_features, self.out_features)) # TO CORRECT
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            if self.group_l0:
                pi = F.sigmoid(self.qz_loga).view(1, self.in_features).expand(batch_size, self.in_features)
            else:
                pi = F.sigmoid(self.qz_loga).view(1, self.in_features, self.out_features).expand(batch_size, self.in_features, self.out_features) # TO CORRECT
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self): # NEW
        if self.group_l0:
            z = self.quantile_concrete(self.get_eps(self.floatTensor(self.in_features)))
        else:
            z = self.quantile_concrete(self.get_eps(self.floatTensor(self.in_features, self.out_features)))

        mask = F.hardtanh(z, min_val=0, max_val=1)
        
        if self.group_l0:
            return mask.view(self.in_features, 1) * self.weights
        else:
            return mask.view(self.in_features, self.out_features) * self.weights

    def get_active_weights(self): # NEW

        return torch.where(self.weights > 0, 1, 0).sum().item()

    def get_mask_active_weights(self): # NEW
        z = self.sample_z(1, False)
        masked_weights = torch.mul(z.T, self.weights.clone().detach())
        
        return torch.where(masked_weights > 0, 1, 0).sum().item()

    def get_mask_fully_active_weights(self): # NEW
        z = self.sample_z(1, False)
        masked_weights = torch.mul(z.T, self.weights.clone().detach())
        
        return torch.where(masked_weights == 1, 1, 0).sum().item()


class L0DisjunctionLayer(nn.Module):
    """The disjunction layer is used to learn the rule sets."""

    def __init__(self, in_features, out_features, random_binarization_rate, use_not=False, bias=False, weight_decay=1., droprate_init=0.5, temperature=2./3., lamba=1., group_l0=False, **kwargs): # UPDATED
        super(L0DisjunctionLayer, self).__init__()
        self.use_not = use_not
        self.node_activation_cnt = None

        self.in_features = in_features if not use_not else in_features * 2 # UPDATED
        self.out_features = out_features # UPDATED
        self.weights = Parameter(0.1 * torch.rand(in_features, out_features)) # UPDATED
        self.randomly_binarize_layer = RandomBinarizationLayer(self.weights.shape, random_binarization_rate) # UPDATED
        self.group_l0 = group_l0
        if self.group_l0:
            self.qz_loga = Parameter(torch.Tensor(in_features)) # NEW
        else:
            self.qz_loga = Parameter(torch.Tensor(in_features, out_features)) # NEW
        self.prior_prec = weight_decay # NEW
        self.temperature = temperature # NEW
        # self.droprate_init = droprate_init if droprate_init != 0. else 0.5 # NEW
        self.droprate_init = droprate_init # NEW
        self.lamba = lamba # NEW
        self.use_bias = False # NEW
        if bias: # NEW
            self.bias = Parameter(torch.Tensor(out_features)) # NEW
            self.use_bias = True # NEW
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor # NEW
        self.reset_parameters() # NEW

    def forward(self, input, randomly_binarize=False): # UPDATED
        if self.use_not:
            input = torch.cat((input, 1 - input), dim=1)
        # if self.local_rep or not self.training: # NEW
        if not self.training: # NEW
            # z = self.sample_z(input.size(0), sample=self.training) # NEW
            z = self.sample_z(1, sample=self.training) # NEW
            # xin = input.mul(z) # NEW
            # processed_input = xin # NEW
            processed_input = input # NEW
            # weights = self.weights # NEW
            weights = z.view(self.in_features, 1) * self.weights # NEW
            # output = xin.mm(self.weights) # NEW
        else:
            weights = self.sample_weights() # NEW
            processed_input = input # NEW
            # output = input.mm(weights) # NEW
        weights = self.randomly_binarize_layer(weights) if randomly_binarize else weights # UPDATED
        output = 1 - torch.prod(1 - processed_input[:, :, None] * weights[None, :, :], dim=1) # UPDATED
        # if self.use_bias: # NEW
        #     output.add_(self.bias) # NEW
        return output

    def binarized_forward(self, x): # UPDATED
        with torch.no_grad():
            if self.use_not:
                x = torch.cat((x, 1 - x), dim=1)
            x = x.type(torch.int)
            z = self.sample_z(1, sample=False) # NEW
            zb = torch.where(z > THRESHOLD_Z, torch.ones_like(z), torch.zeros_like(z)).type(torch.int) # NEW
            # xin = x.mul(zb) # NEW
            # processed_input = xin # NEW
            processed_input = x
            Wb = zb.view(self.in_features, 1) * torch.where(self.weights > THRESHOLD_W, torch.ones_like(self.weights), torch.zeros_like(self.weights)).type(torch.int) # UPDATED
            # Wb = torch.where(self.qz_loga > 0.0, torch.ones_like(self.qz_loga), torch.zeros_like(self.qz_loga)).type(torch.int) # UPDATED
            return 1 - torch.prod(1 - processed_input[:, :, None] * Wb[None, :, :], dim=1) # UPDATED

    def reset_parameters(self): # NEW
        # init.kaiming_normal(self.weights, mode='fan_out')

        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs): # NEW
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x): # NEW
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def one_minus_cdf_qz0(self):
        # let's use what is described in the paper (it's more specific, while the above is more general)
        # note: in the paper they call qz_loga -> logaj (better check to make sure i'm not misunderstanding)
        return F.sigmoid(self.qz_loga - self.temperature * math.log(-limit_a / limit_b))

    def quantile_concrete(self, x): # NEW
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self): # NEW
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        if self.group_l0:
            logpw_col = torch.sum(- (.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 1)
            logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
            logpb = 0 if not self.use_bias else - torch.sum(.5 * self.prior_prec * self.bias.pow(2))
            return logpw + logpb
        else:
            lc = torch.sum(self.one_minus_cdf_qz0())
            return lc

    def regularization(self): # NEW
        if self.group_l0:
            return self._reg_w()
        else:
            return self._reg_w() * self.lamba

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.out_features
        expected_l0 = ppos * self.out_features
        if self.use_bias:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        return expected_flops.item(), expected_l0.item()

    def get_eps(self, size): # NEW
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True): # NEW
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(self.floatTensor(batch_size, self.in_features))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = F.sigmoid(self.qz_loga).view(1, self.in_features).expand(batch_size, self.in_features)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self): # NEW
        if self.group_l0:
            z = self.quantile_concrete(self.get_eps(self.floatTensor(self.in_features)))
        else:
            z = self.quantile_concrete(self.get_eps(self.floatTensor(self.in_features, self.out_features)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        
        if self.group_l0:
            return mask.view(self.in_features, 1) * self.weights
        else:
            return mask.view(self.in_features, self.out_features) * self.weights

    def get_active_weights(self): # NEW
        return torch.where(self.weights > 0, 1, 0).sum().item()

    def get_mask_active_weights(self): # NEW
        z = self.sample_z(1, False)
        masked_weights = torch.mul(z.T, self.weights.clone().detach())
        
        return torch.where(masked_weights > 0, 1, 0).sum().item()

    def get_mask_fully_active_weights(self): # NEW
        z = self.sample_z(1, False)
        masked_weights = torch.mul(z.T, self.weights.clone().detach())
        
        return torch.where(masked_weights == 1, 1, 0).sum().item()



class L0MLLP(nn.Module):
    """The Multilayer Logical Perceptron (MLLP) used for Concept Rule Sets (CRS) learning.

    For more information, please read our paper: Transparent Classification with Multilayer Logical Perceptrons and
    Random Binarization."""


    def __init__(self, dim_list, device, random_binarization_rate=0.75, use_not=False, log_file=None, N=50000, beta_ema=0.999,
                 weight_decay=1, lamba=0.1, droprate_init_input=0.2, droprate_init=0.5, temperature=2./3., group_l0=False, use_bias=False): # UPDATED
        """

        Parameters
        ----------
        dim_list : list
            A list specifies the number of nodes (neurons) of all the layers in MLLP from bottom to top. dim_list[0]
            should be the dimensionality of the input data and dim_list[1] should be the number of class labels.
        device : torch.device
            Run on which device.
        random_binarization_rate : float
            The rate of the random binarization in the Random Binarizatoin (RB) method. RB method is important for CRS
            extractions from deep MLLPs.
        use_not : bool
            Whether use the NOT (~) operator in logical rules.
        log_file : str
            The path of the log file. If log_file is None, use sys.stdout as the output stream.
        """

        super(L0MLLP, self).__init__()

        log_format = '[%(levelname)s] - %(message)s'
        if log_file is None:
            logging.basicConfig(level=logging.INFO, stream=sys.stdout, format=log_format)
        else:
            logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format=log_format)

        self.random_binarization_rate = random_binarization_rate
        self.N = N # NEW
        self.beta_ema = beta_ema # NEW
        self.weight_decay = self.N * weight_decay # NEW
        # self.weight_decay = weight_decay # NEW
        self.lamba = lamba # NEW
        self.temperature = temperature # NEW
        self.droprate_init_input = droprate_init_input # NEW
        self.droprate_init = droprate_init # NEW
        self.dim_list = dim_list
        self.device = device
        self.use_not = use_not
        self.use_bias = use_bias # NEW
        self.group_l0 = group_l0 # NEW
        self.enc = None
        self.conj = []
        self.disj = []

        print(f"N value: '{N}'")

        for i in range(0, len(dim_list) - 2, 2):
            conj = L0ConjunctionLayer(dim_list[i], dim_list[i+1], self.random_binarization_rate, use_not=self.use_not, droprate_init=self.droprate_init_input if i == 0 else self.droprate_init, weight_decay=self.weight_decay,
                               lamba=self.lamba, temperature=self.temperature, bias=self.use_bias, group_l0=self.group_l0)
            disj = L0DisjunctionLayer(dim_list[i + 1], dim_list[i + 2], self.random_binarization_rate, use_not=False, droprate_init=self.droprate_init, weight_decay=self.weight_decay,
                               lamba=self.lamba, temperature=self.temperature, bias=self.use_bias, group_l0=self.group_l0)
            self.add_module('conj{}'.format(i), conj)
            self.add_module('disj{}'.format(i), disj)
            self.conj.append(conj)
            self.disj.append(disj)

        self.layers = self.conj + self.disj # NEW

        if beta_ema > 0.: # NEW
            print('Using temporal averaging with beta: {}'.format(beta_ema)) # NEW
            self.avg_param = deepcopy(list(p.data for p in self.parameters())) # NEW
            if torch.cuda.is_available(): # NEW
                self.avg_param = [a.cuda() for a in self.avg_param] # NEW
            self.steps_ema = 0. # NEW

    def forward(self, x, randomly_binarize=False): # UPDATED
        for conj, disj in zip(self.conj, self.disj):
            x = conj(x, randomly_binarize=randomly_binarize) # UPDATED
            x = disj(x, randomly_binarize=randomly_binarize) # UPDATED
        return x

    def binarized_forward(self, x):
        """Equivalent to using the extracted Concept Rule Sets."""
        with torch.no_grad():
            for conj, disj in zip(self.conj, self.disj):
                x = conj.binarized_forward(x)
                x = disj.binarized_forward(x)
        return x

    def get_active_weights(self): # NEW
        active_weights = 0
        for conj, disj in zip(self.conj, self.disj):
            active_weights += conj.get_active_weights()
            active_weights += disj.get_active_weights()
        return active_weights

    def get_mask_active_weights(self): # NEW
        mask_active_weights = 0
        for conj, disj in zip(self.conj, self.disj):
            mask_active_weights += conj.get_mask_active_weights()
            mask_active_weights += disj.get_mask_active_weights()
        return mask_active_weights

    def get_mask_fully_active_weights(self): # NEW
        mask_fully_active_weights = 0
        for conj, disj in zip(self.conj, self.disj):
            mask_fully_active_weights += conj.get_mask_fully_active_weights()
            mask_fully_active_weights += disj.get_mask_fully_active_weights()
        return mask_fully_active_weights

    def get_total_weights(self): # NEW
        total_weights = 0
        for layer in self.layers:
            total_weights += (layer.weights.size()[0] * layer.weights.size()[1])
        
        return total_weights
        
    def clip(self):
        """Clip the weights into the range [0, 1]."""
        for name, param in self.named_parameters():
            if "weights" in name:
                param.data.clamp_(0, 1)

    def randomly_binarize_layer_refresh(self):
        """Change the set of weights to be binarized."""
        for conj, disj in zip(self.conj, self.disj):
            conj.randomly_binarize_layer.refresh()
            disj.randomly_binarize_layer.refresh()

    def data_transform(self, X, y):
        X = X.astype(np.float32)
        if y is None:
            return torch.tensor(X)
        y = y.astype(np.float32)
        logging.debug('{}'.format(y.shape))
        logging.debug('{}'.format(y[:20]))
        return torch.tensor(X), torch.tensor(y)  # Do not put all the data in GPU at once.

    def regularization(self): # NEW
        regularization = 0.
        for layer in self.layers:
            if self.group_l0:
                regularization += - (1. / self.N) * layer.regularization()
            else:
                regularization += layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self): # NEW
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self): # NEW
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self): # NEW
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params): # NEW
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self): # NEW
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        if epoch % lr_decay_epoch == 0:
            logging.info('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train_model(self, X=None, y=None, X_validation=None, y_validation=None, data_loader=None, epoch=50, lr=0.01, lr_decay_epoch=100,
              lr_decay_rate=0.75, batch_size=64): # UPDATED
        """

        Parameters
        ----------
        X : numpy.ndarray, shape = [n_samples, n_features]
            The training input instances. All the values should be 0 or 1.
        y : numpy.ndarray, shape = [n_samples, n_classes]
            The class labels. All the values should be 0 or 1.
        X_validation : numpy.ndarray, shape = [n_samples, n_features]
            The input instances of validation set. The format of X_validation is the same as X.
            if X_validation is None, use the training set (X) for validation.
        y_validation : numpy.ndarray, shape = [n_samples, n_classes]
            The class labels of validation set. The format of y_validation is the same as y.
            if y_validation is None, use the training set (y) for validation.
        epoch : int
            The total number of epochs during the training.
        lr : float
            The initial learning rate.
        lr_decay_epoch : int
            Decay learning rate every lr_decay_epoch epochs.
        lr_decay_rate : float
            Decay learning rate by a factor of lr_decay_rate.
        batch_size : int
            The batch size for training.
        weight_decay : float
            The weight decay (L2 penalty).

        Returns
        -------
        loss_log : list
            Training loss of MLLP during the training.
        accuracy : list
            Accuracy of MLLP on the validation set during the training.
        accuracy_b : list
            Accuracy of CRS on the validation set during the training.
        f1_score : list
            F1 score (Macro) of MLLP on the validation set during the training.
        f1_score_b : list
            F1 score (Macro) of CRS on the validation set during the training.

        """

        torch.autograd.set_detect_anomaly(True)

        self.train()

        if (X is None or y is None) and data_loader is None:
            raise Exception("Both data set and data loader are unavailable.")
        if data_loader is None:
            X, y = self.data_transform(X, y)
            if X_validation is not None and y_validation is not None:
                X_validation, y_validation = self.data_transform(X_validation, y_validation)
            data_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        loss_log = []
        accuracy = []
        accuracy_b = []
        f1_score = []
        f1_score_b = []
        accuracy_v = [] if X_validation is not None and y_validation is not None else None
        accuracy_v_b = [] if X_validation is not None and y_validation is not None else None
        f1_score_v = [] if X_validation is not None and y_validation is not None else None
        f1_score_v_b = [] if X_validation is not None and y_validation is not None else None
        total_mask_active_weights_list = []
        total_active_weights_list = []
        total_mask_fully_active_weights_list = []

        # criterion = nn.BCELoss()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr) # UPDATED

        # define loss function (criterion) and optimizer
        def loss_function(output, target_var): # NEW
            loss = criterion(output, target_var)
            total_loss = loss + self.regularization()
            if torch.cuda.is_available():
                total_loss = total_loss.to(self.device)
            return total_loss

        # Get total weights
        total_weights = self.get_total_weights()

        for epo in tqdm(range(epoch), desc="Epochs"):

            optimizer = self.exp_lr_scheduler(optimizer, epo, init_lr=lr, lr_decay_rate=lr_decay_rate,
                                              lr_decay_epoch=lr_decay_epoch)
            
            num_examples = 0
            running_loss = 0.0
            cnt = 0
            for X, y in data_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()  # Zero the gradient buffers.
                y_pred = self.forward(X, randomly_binarize=True if self.random_binarization_rate > 0.0 else False)
                loss = loss_function(y_pred, y) # UPDATED
                running_loss += (loss.item() * X.size(0))
                num_examples += X.size(0)
                loss.backward()
                if epo % 100 == 0 and cnt == 0:
                    for param in self.parameters():
                        logging.debug('{}'.format(param.grad))
                    cnt += 1
                optimizer.step()

                for k, layer in enumerate(self.layers):  # NEW
                    layer.constrain_parameters() # NEW

                if self.beta_ema > 0.: # NEW
                    self.update_ema() # NEW

                self.clip()
            
            self.randomly_binarize_layer_refresh()

            # Get train mask active weights and active weights
            total_mask_active_weights = self.get_mask_active_weights() # NEW
            total_active_weights = self.get_active_weights() # NEW
            total_mask_fully_active_weights = self.get_mask_fully_active_weights() # NEW
            print(f"Total mask active weights: '{total_mask_active_weights}'")


            logging.info('epoch: {}, loss: {}'.format(epo, running_loss / num_examples))
            print('epoch: {}, loss: {}'.format(epo, running_loss / num_examples))
            loss_log.append(running_loss / num_examples)
            total_mask_active_weights_list.append(total_mask_active_weights) # NEW
            total_active_weights_list.append(total_active_weights) # NEW
            total_mask_fully_active_weights_list.append(total_mask_fully_active_weights) # NEW
            
            # Change the set of weights to be binarized every epoch (Random Binarization)

            # Test the validation set or training set every 5 epochs.
            if epo % 5 == 0:
                if X_validation is not None and y_validation is not None:
                    acc_v, acc_v_b, f1_v, f1_v_b = self.test(X_validation, y_validation, False)
                    set_name = 'Validation'
                else:
                    acc_v, acc_v_b, f1_v, f1_v_b = (None, None, None, None)
                acc, acc_b, f1, f1_b = self.test(X, y, False)
                set_name = 'Training'
                logging.info('-' * 60)
                logging.info('On {} Set:\n\tAccuracy of MLLP Model: {}'
                             '\n\tAccuracy of CRS  Model: {}'.format(set_name, acc, acc_b))
                logging.info('On {} Set:\n\tF1 Score of MLLP Model: {}'
                             '\n\tF1 Score of CRS  Model: {}'.format(set_name, f1, f1_b))
                logging.info('-' * 60)
                accuracy.append(acc)
                accuracy_b.append(acc_b)
                f1_score.append(f1)
                f1_score_b.append(f1_b)
                if X_validation is not None and y_validation is not None:
                    accuracy_v.append(acc_v)
                    accuracy_v_b.append(acc_v_b)
                    f1_score_v.append(f1_v)
                    f1_score_v_b.append(f1_v_b)
        return loss_log, accuracy, accuracy_b, f1_score, f1_score_b, accuracy_v, accuracy_v_b, f1_score_v, f1_score_v_b, total_mask_active_weights_list, total_active_weights_list, total_mask_fully_active_weights_list, total_weights

    def test(self, X, y, need_transform=True):
        if need_transform:
            X, y = self.data_transform(X, y)
        
        self.eval()
        with torch.no_grad():
            if self.beta_ema > 0:
                old_params = self.get_params()
                self.load_ema_params()

            X = X.to(self.device)
            test_loader = DataLoader(TensorDataset(X), batch_size=128, shuffle=False)

            y = y.cpu().numpy().astype(int)
            y = np.argmax(y, axis=1)
            data_num = y.shape[0]
            slice_step = data_num // 40 if data_num >= 40 else 1
            logging.debug('{} {}'.format(y.shape, y[:: slice_step]))

            # Test the model batch by batch.
            # Test the MLLP.
            y_pred_list = []
            for X, in test_loader:
                y_pred_list.append(self.forward(X))
            y_pred = torch.cat(y_pred_list)
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            logging.debug('{} {}'.format(y_pred.shape, y_pred[:: slice_step]))

            # Test the CRS.
            y_pred_b_list = []
            for X, in test_loader:
                y_pred_b_list.append(self.binarized_forward(X))
            y_pred_b = torch.cat(y_pred_b_list)
            y_pred_b = y_pred_b.cpu().numpy()
            logging.debug('y_pred_b: {} {}'.format(y_pred_b.shape, y_pred_b[:: (slice_step * 2)]))
            y_pred_b = np.argmax(y_pred_b, axis=1)
            logging.debug('{} {}'.format(y_pred_b.shape, y_pred_b[:: slice_step]))

            accuracy = metrics.accuracy_score(y, y_pred)
            accuracy_b = metrics.accuracy_score(y, y_pred_b)

            f1_score = metrics.f1_score(y, y_pred, average='macro')
            f1_score_b = metrics.f1_score(y, y_pred_b, average='macro')

            if self.beta_ema > 0:
                self.load_params(old_params)
        return accuracy, accuracy_b, f1_score, f1_score_b

    def detect_dead_node(self, X, need_transform=True):
        if need_transform:
            X = self.data_transform(X, None)
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(X), batch_size=128, shuffle=False)
            for conj, disj in zip(self.conj, self.disj):
                conj.node_activation_cnt = torch.zeros(conj.out_features, dtype=torch.long, device=self.device)
                disj.node_activation_cnt = torch.zeros(disj.out_features, dtype=torch.long, device=self.device)

            # Test the model batch by batch.
            for x, in test_loader:
                x = x.to(self.device)
                for conj, disj in zip(self.conj, self.disj):
                    x = conj.binarized_forward(x)
                    conj.node_activation_cnt += torch.sum(x, dim=0)
                    x = disj.binarized_forward(x)
                    disj.node_activation_cnt += torch.sum(x, dim=0)

    def get_rules(self, X=None):
        """Extract rules from parameters of MLLP."""
        # If X is not None, detect the dead nodes using X.
        if X is not None:
            self.detect_dead_node(X)
            activation_cnt_list = [np.sum(np.concatenate([X, 1 - X], axis=1) if self.use_not else X, axis=0)]
            for conj, disj in zip(self.conj, self.disj):
                activation_cnt_list.append(conj.node_activation_cnt.cpu().numpy())
                activation_cnt_list.append(disj.node_activation_cnt.cpu().numpy())
        else:
            activation_cnt_list = None

        # Get the rules from the top layer to the bottom layer.
        param_list = list(param for name, param in self.named_parameters() if "weights" in name) # UPDATED
        z_list = []
        for conj, disj in zip(self.conj, self.disj): # NEW
            z_list.append(conj.sample_z(1, sample=False))
            z_list.append(disj.sample_z(1, sample=False))
        n_param = len(param_list)
        mark = {}
        rules_list = []
        for i in reversed(range(n_param)):
            param = param_list[i]
            W = param.T.cpu().detach().numpy() # UPDATED
            z = z_list[i][0].cpu().detach().numpy() # NEW
            rules = defaultdict(list)
            num = self.dim_list[i]
            for k, row in enumerate(W):
                if i != n_param - 1 and ((i, k) not in mark):
                    continue
                if X is not None and activation_cnt_list[i + 1][k] < 1:
                    continue
                found = False
                for j, (wj, zj) in enumerate(zip(row, z)):
                    if X is not None and activation_cnt_list[i][j % num] < 1:
                        continue
                    if wj > THRESHOLD_W and zj > THRESHOLD_Z:
                        rules[k].append(j)
                        mark[(i - 1, j % num)] = 1
                        found = True
                if not found:
                    rules[k] = []
            rules_list.append(rules)
        return rules_list

    def eliminate_redundant_rules(self, rules_list):
        """Eliminate redundant rules to simplify the extracted CRS."""
        rules_list = copy.deepcopy(rules_list)
        for i in reversed(range(len(rules_list))):
            # Eliminate the redundant part of each rule from bottom to top.
            if i != len(rules_list) - 1:
                num = self.dim_list[len(self.dim_list) - i - 2]
                for k, v in rules_list[i].items():
                    mark = {}
                    new_rule = []
                    for j1 in range(len(v)):
                        if j1 in mark:
                            continue
                        for j2 in range(j1 + 1, len(v)):
                            if j2 in mark:
                                continue
                            if j1 // num != j2 // num:
                                continue
                            s1 = set(rules_list[i + 1][v[j1 % num]])
                            s2 = set(rules_list[i + 1][v[j2 % num]])
                            if s1.issuperset(s2):
                                mark[j1] = 1
                                break
                            elif s1.issubset(s2):
                                mark[j2] = 1
                        if j1 not in mark:
                            new_rule.append(v[j1])
                    rules_list[i][k] = sorted(list(set(new_rule)))

            # Merge the identical nodes.
            union_find = UnionFind(rules_list[i].keys())
            kv_list = list(rules_list[i].items())
            n_kv = len(kv_list)
            if i > 0:
                for j1 in range(n_kv):
                    k1, v1 = kv_list[j1]
                    for j2 in range(j1 + 1, n_kv):
                        k2, v2 = kv_list[j2]
                        if v1 == v2:
                            union_find.union(k1, k2)
                # Update the upper layer.
                for k, v in rules_list[i - 1].items():
                    for j in range(len(v)):
                        v[j] = union_find.find(v[j])
                    rules_list[i - 1][k] = sorted(list(set(v)))
        # Get the final simplified rules.
        new_rules_list = []
        mark = {}
        for i in range(len(rules_list)):
            num = self.dim_list[len(self.dim_list) - i - 2]
            rules = defaultdict(list)
            for k, v in rules_list[i].items():
                if i != 0 and ((i, k) not in mark):
                    continue
                for j in v:
                    mark[(i + 1, j % num)] = 1
                    rules[k].append(j)
            new_rules_list.append(rules)
        return new_rules_list

    def get_name(self, i, j, X_fname=None, y_fname=None):
        nl = len(self.dim_list)
        num = self.dim_list[nl - i - 1]
        if j >= num:
            j -= num
            prefix = '~'
        else:
            prefix = ' '
        if X_fname is not None and i == nl - 1:
            name = X_fname[j]
        elif y_fname is not None and i == 0:
            name = y_fname[j]
        else:
            name = '{}{},{}'.format('s' if i % 2 == 0 else 'r', (nl - 2 - i) // 2 + 1, j)
        name = prefix + name
        return name

    def concept_rule_set_print(self, X=None, X_fname=None, y_fname=None, file=sys.stdout, eliminate_redundancy=True):
        """Print the Concept Rule Sets extracted from the trained Multilayer Logical Perceptron."""
        if eliminate_redundancy:
            rules_list = self.eliminate_redundant_rules(self.get_rules(X))
        else:
            rules_list = self.get_rules(X)
        for i in range(0, len(rules_list), 2):
            rules_str = defaultdict(list)
            for k, v in rules_list[i + 1].items():
                for j in v:
                    rules_str[k].append(self.get_name(i + 2, j, X_fname=X_fname, y_fname=y_fname))
            rule_sets = defaultdict(list)
            num = self.dim_list[len(self.dim_list) - i - 2]
            for k, v in rules_list[i].items():
                for j in v:
                    if j >= num:
                        jn = j - num
                        prefix = '~'
                    else:
                        prefix = ' '
                        jn = j
                    rule_sets[self.get_name(i, k, X_fname=X_fname, y_fname=y_fname)].append(
                        '{:>10}:\t{}{}'.format(self.get_name(i + 1, j, X_fname=X_fname, y_fname=y_fname), prefix,
                                               rules_str[jn]))
            print('-' * 90, file=file)
            for k, v in rule_sets.items():
                print('{}:'.format(k), file=file)
                for r in v:
                    print('\t', r, file=file)

        return rules_list