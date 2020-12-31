"""
Core Harmonic Convolution Implementation
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def h_conv(X, W, strides=(1,1,1,1), padding=0, max_order=1):
    """Inter-order (cross-stream) convolutions can be implemented as single
    convolution. For this we store data as 6D tensors and filters as 8D
    tensors, at convolution, we reshape down to 4D tensors and expand again.

    X: tensor shape [mbatch,h,w,order,complex,channels]
    Q: tensor dict---reshaped to [h,w,in,in.comp,in.ord,out,out.comp,out.ord]
    P: tensor dict---phases
    strides: as per tf convention (default (1,1,1,1))
    padding: as per tf convention (default VALID)
    filter_size: (default 3)
    max_order: (default 1)
    name: (default h_conv)
    """	
    Xsh = list(X.size())
    X_ = X.view(Xsh[:3]+[-1]) #flatten out the last 3 dimensions

    # The script below constructs the stream-convolutions as one big filter
    # W_. For each output order, run through each input order and
    # copy-paste the filter for that convolution.

    W_ = []
    for output_order in range(max_order+1):
        # For each output order build input
        Wr = []
        Wi = []
        for input_order in range(Xsh[3]):
            # Difference in orders is the convolution order
            weight_order = output_order - input_order
            weights = W[np.abs(weight_order)]
            sign = np.sign(weight_order)
            # Choose a different filter depending on whether input is real.
            # We have the arbitrary convention that negative orders use the
            # conjugate weights.
            if Xsh[4] == 2:
                Wr += [weights[0],-sign*weights[1]]
                Wi += [sign*weights[1],weights[0]]
            else:
                Wr += [weights[0]]
                Wi += [weights[1]]

        W_ += [torch.cat(Wr, 2), torch.cat(Wi, 2)]
    W_ = torch.cat(W_, 3)

    # Convolve
    W_ = W_.permute(3, 2, 0, 1)
    W_ = W_.type(torch.cuda.FloatTensor) if torch.cuda.is_available() \
                                            else W_.type(torch.FloatTensor)
    
    X_ = X_.permute(0, 3, 1, 2)
    Y = torch.nn.functional.conv2d(X_, W_, stride=strides, padding=padding)
    Y = Y.permute(0, 2, 3, 1)
    # Reshae results into appropriate format
    Ysh = list(Y.size())
    new_shape = Ysh[:3] + [max_order+1,2] + [Ysh[3]//(2*(max_order+1))]
    return Y.view(*new_shape)


def mean_pooling(x, ksize=(1,1), strides=(1,1)):
    """Implement mean pooling on complex-valued feature maps. The complex mean
    on a local receptive field, is performed as mean(real) + i*mean(imag)

    x: tensor shape [mbatch,h,w,order,complex,channels]
    ksize: kernel size 4-tuple (default (1,1,1,1))
    strides: stride size 4-tuple (default (1,1,1,1))
    """
    Xsh = list(x.size())
    # Collapse output the order, complex, and channel dimensions
    X_ = x.view(*(Xsh[:3]+[-1]))
    X_ = X_.permute(0, 3, 1, 2)
    Y = F.avg_pool2d(X_, kernel_size=ksize, stride=strides, padding=0)
    Y = Y.permute(0, 2, 3, 1)
    Ysh = list(Y.size())
    new_shape = Ysh[:3] + Xsh[3:]
    return Y.view(*new_shape)


def stack_magnitudes(X, eps=1e-12, keep_dims=True):
    """Stack the magnitudes of each of the complex feature maps in X.

    Output U = concat(|X_i|)

    X: dict of channels {rotation order: (real, imaginary)}
    eps: regularization since grad |Z| is infinite at zero (default 1e-12)
    """
    R = torch.sum(torch.mul(X, X), dim=(4,), keepdim=keep_dims)
    return torch.sqrt(torch.clamp(R,min=eps))

	

##### FUNCTIONS TO CONSTRUCT STEERABLE FILTERS #####
def get_interpolation_weights(filter_size, m, n_rings=None):
    """Resample the patches on rings using Gaussian interpolation"""
    if n_rings is None:
        n_rings = np.maximum(filter_size/2, 2)
    radii = np.linspace(m!=0, n_rings-0.5, n_rings) #<-------------------------look into m and n-rings-0.5
    # We define pixel centers to be at positions 0.5
    foveal_center = np.asarray([filter_size, filter_size])/2.
    # The angles to sample
    N = n_samples(filter_size)
    lin = (2*np.pi*np.arange(N))/N
    # Sample equi-angularly along each ring
    ring_locations = np.vstack([-np.sin(lin), np.cos(lin)])
    # Create interpolation coefficient coordinates
    coords = L2_grid(foveal_center, filter_size)
    # Sample positions wrt patch center IJ-coords
    radii = radii[:,np.newaxis,np.newaxis,np.newaxis]
    ring_locations = ring_locations[np.newaxis,:,:,np.newaxis]
    diff = radii*ring_locations - coords[np.newaxis,:,np.newaxis,:]
    dist2 = np.sum(diff**2, axis=1)
    # Convert distances to weightings
    bandwidth = 0.5
    weights = np.exp(-0.5*dist2/(bandwidth**2))
    # Normalize
    return weights/np.sum(weights, axis=2, keepdims=True)


def get_filters(R, filter_size, P=None, n_rings=None):
    """Perform single-frequency DFT on each ring of a polar-resampled patch"""
       
    k = filter_size
    filters = {}
    N = n_samples(k)
    from scipy.linalg import dft
    for m, r in R.items():
        rsh = list(r.size())
        # Get the basis matrices
        weights = get_interpolation_weights(k, m, n_rings=n_rings)
        DFT = dft(N)[m,:]
        LPF = np.dot(DFT, weights).T

        cosine = np.real(LPF).astype(np.float32)
        sine = np.imag(LPF).astype(np.float32)
        # Reshape for multiplication with radial profile
        cosine = torch.from_numpy(cosine)
        cosine = cosine.to(device="cuda" if torch.cuda.is_available() else "cpu")
        sine = torch.from_numpy(sine)
        sine = sine.to(device="cuda" if torch.cuda.is_available() else "cpu")
        # Project taps on to rotational basis
        r = r.view(rsh[0],rsh[1]*rsh[2])
        ucos = torch.matmul(cosine, r).view(k, k, rsh[1], rsh[2]).double()
        usin = torch.matmul(sine, r).view(k, k, rsh[1], rsh[2]).double()
        if P is not None:
            # Rotate basis matrices
            ucos_ = torch.cos(P[m])*ucos + torch.sin(P[m])*usin
            usin = -torch.sin(P[m])*ucos + torch.cos(P[m])*usin
            ucos = ucos_
        filters[m] = (ucos, usin)

    return filters


def n_samples(filter_size):
    return np.maximum(np.ceil(np.pi*filter_size),101) ############## <--- One source of instability


def L2_grid(center, shape):
    # Get neighbourhoods
    lin = np.arange(shape)+0.5
    J, I = np.meshgrid(lin, lin)
    I = I - center[1]
    J = J - center[0]
    return np.vstack((np.reshape(I, -1), np.reshape(J, -1)))


























