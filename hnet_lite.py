"""
Harmonic Convolutions Lite

A simplified API for harmomin_network_ops
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hnet_ops import *
from hnet_ops import h_conv

class Conv2d(nn.Module):

    @staticmethod
    def get_weights(filter_shape, W_init=None, std_mult=0.4):
        """Initialize weights variable with He method

        filter_shape: list of filter dimensions
        W_init: numpy initial values (default None)
        std_mult: multiplier for weight standard deviation (default 0.4)
        name: (default W)
        device: (default /cpu:0)
        """

        if W_init == None:
            stddev = std_mult*np.sqrt(2.0 / np.prod(filter_shape[:3]))
            W_init = torch.normal(torch.zeros(*filter_shape), stddev)
                    
        return nn.Parameter(W_init)

    @staticmethod
    def init_weights_dict(shape, max_order, std_mult=0.4, n_rings=None):
        """
        Initializes the dict of weights

        """
        if isinstance(max_order, int):
            orders = range(-max_order, max_order+1)
        else:
            diff = max_order[1]-max_order[0]
            orders = range(-diff, diff+1)
        weights_dict = {}
        for i in orders:
            if n_rings is None:
                n_rings = np.maximum(shape[0]/2, 2)
            sh = [n_rings,] + list(shape[2:])
            weights_dict[i] = Conv2d.get_weights(sh, std_mult=std_mult)
        return weights_dict

    @staticmethod
    def init_phase_dict(n_in, n_out, max_order):
        """Return a dict of phase offsets"""
        if isinstance(max_order, int):
            orders = range(-max_order, max_order+1)
        else:
            diff = max_order[1]-max_order[0]
            orders = range(-diff, diff+1)
        phase_dict = {}
        for i in orders:
            init = np.random.rand(1,1,n_in,n_out) * 2. *np.pi
            init = torch.from_numpy(init)
            phase = nn.Parameter(init)
            phase_dict[i] = phase
        return phase_dict

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, phase=True,
             max_order=1, stddev=0.4, n_rings=None):
        super().__init__()
        """Harmonic Convolution lite

        x: input tf tensor, shape [batchsize,height,width,order,complex,channels],
        e.g. a real input tensor of rotation order 0 could have shape
        [16,32,32,3,1,9], or a complex input tensor of rotation orders 0,1,2, could
        have shape [32,121,121,3,2,10]
        in_channels: Number of input channels (int)
        out_channels: Number of output channels (int)
        kernel_size: size of square filter (int)
        stride: stride size (4-tuple: default (1,1,1,1))
        padding: SAME or VALID (defult VALID)
        phase: use a per-channel phase offset (default True)
        max_order: maximum rotation order e.g. max_order=2 uses 0,1,2 (default 1)
        stddev: scale of filter initialization wrt He initialization
        name: (default 'lconv')
        device: (default '/cpu:0')
        """ 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.phase = phase
        self.max_order = max_order
        self.stddev = stddev
        self.n_rings = n_rings
        self.shape = (kernel_size, kernel_size, self.in_channels, self.out_channels)
        self.Q = Conv2d.init_weights_dict(self.shape, self.max_order, self.stddev, self.n_rings)
        for k, v in self.Q.items():
            self.register_parameter('weights_dict_' + str(k), v)
        if self.phase:
            self.P = Conv2d.init_phase_dict(self.in_channels, self.out_channels, self.max_order)
            for k, v in self.P.items():
                self.register_parameter('phase_dict_' + str(k), v)
        else:
            self.P = None

    def forward(self, x):
        W = get_filters(self.Q, filter_size=self.kernel_size, P=self.P, n_rings=self.n_rings)
        R = h_conv(x, W, strides=self.stride, padding=self.padding, max_order=self.max_order)
        return R

class HNonlin(nn.Module):
    """Apply the nonlinearity described by the function handle fnc: R -> R+ to
    the magnitude of X. CAVEAT: fnc must map to the non-negative reals R+.

    Output U + iV = fnc(R+b) * (A+iB)
    where  A + iB = Z/|Z|

    X: dict of channels {rotation order: (real, imaginary)}
    fnc: function handle for a nonlinearity. MUST map to non-negative reals R+
    eps: regularization since grad |Z| is infinite at zero (default 1e-8)
    """

    def __init__(self, fnc, order, channels, eps=1e-12):
        super().__init__()
        self.fnc = fnc
        self.eps = eps
        self.b = nn.Parameter(torch.FloatTensor(1,1,1,order,1,channels))
        nn.init.xavier_normal_(self.b)

    def forward(self, X):	
        magnitude = stack_magnitudes(X, self.eps)
        Rb = magnitude + self.b
        c = self.fnc(Rb) / magnitude
        return c * X


class BatchNorm(nn.Module):

    def __init__(self, order, cmplx, channels, fnc=F.relu, decay=0.99, eps=1e-4):
        super().__init__()
        self.fnc = fnc
        self.eps = eps
        self.n_out = order,cmplx,channels
        self.tn_out = order*cmplx*channels
        self.bn = nn.BatchNorm1d(self.tn_out, eps=self.eps, momentum=1-decay)

    def forward(self, X: torch.Tensor):
        magnitude = stack_magnitudes(X, self.eps)
        Xsh = tuple(X.size())
        assert Xsh[-3:]==self.n_out, (Xsh, self.n_out)
        X = X.reshape(-1, self.tn_out)
        Rb = self.bn(X)
        X = X.view(Xsh)
        Rb = Rb.view(Xsh)
        c = self.fnc(Rb) / magnitude
        return c*X


def mean_pool(x, ksize=(1,1), strides=(1,1)):
    """Mean pooling"""
    return mean_pooling(x, ksize=ksize, strides=strides)


def sum_magnitudes(x, eps=1e-12, keep_dims=True):
    """Sum the magnitudes of each of the complex feature maps in X.

    Output U = sum_i |x_i|

    x: input tf tensor, shape [batchsize,height,width,channels,complex,order],
    e.g. a real input tensor of rotation order 0 could have shape
    [16,32,32,3,1,1], or a complex input tensor of rotation orders 0,1,2, could
    have shape [32,121,121,32,2,3]
    eps: regularization since grad |x| is infinite at zero (default 1e-4)
    keep_dims: whether to collapse summed dimensions (default True)
    """
    R = torch.sum(torch.mul(x,x), dim=(4,), keepdim=keep_dims)
    return torch.sum(torch.sqrt(torch.clamp(R,eps)), dim=(3,), keepdim=keep_dims)


