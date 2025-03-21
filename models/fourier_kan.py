import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import *

# code modified from https://github.com/GistNoesis/FourierKAN/
# https://github.com/IcurasLW/FR-KAN/blob/main/src/efficient_kan/fourier_kan.py

#This is inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients
#It should be easier to optimize as fourier are more dense than spline (global vs local)
#Once convergence is reached you can replace the 1d function with spline approximation for faster evaluation giving almost the same result
#The other advantage of using fourier over spline is that the function are periodic, and therefore more numerically bounded
#Avoiding the issues of going out of grid

class FourierKANLayer(nn.Module):
    def __init__( self, inputdim, outdim, gridsize,addbias=True):
        super(FourierKANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        
        self.layernorm = nn.LayerNorm(inputdim)
        
        #The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        #then each coordinates of the output is of unit variance 
        #independently of the various sizes
        self.fouriercoeffs = torch.nn.Parameter(torch.randn(2,outdim,inputdim,gridsize) / 
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize) ) )
        if self.addbias:
            self.bias  = torch.nn.Parameter(torch.zeros(1,outdim))

    #x.shape ( ... , indim ) 
    #out.shape ( ..., outdim)
    def forward(self,x):
        
        x = self.layernorm(x) # layer norm
         
        xshp = x.shape
        outshape = xshp[0:-1]+(self.outdim,)
        x = torch.reshape(x,(-1,self.inputdim))
        #Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1,self.gridsize+1,device=x.device),(1,1,1,self.gridsize))
        xrshp = torch.reshape(x,(x.shape[0],1,x.shape[1],1))
        #This should be fused to avoid materializing memory
        c = torch.cos( k*xrshp )
        s = torch.sin( k*xrshp )
        #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them 
        y =  torch.sum( c*self.fouriercoeffs[0:1],(-2,-1)) 
        y += torch.sum( s*self.fouriercoeffs[1:2],(-2,-1))
        if( self.addbias):
            y += self.bias
        #End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = th.reshape(c,(1,x.shape[0],x.shape[1],self.gridsize))
        s = th.reshape(s,(1,x.shape[0],x.shape[1],self.gridsize))
        y2 = th.einsum( "dbik,djik->bj", th.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = torch.reshape(y, outshape)
        return y

class FourierKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 8,
        spline_order: int = 0, #  placeholder
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FourierKANLayer(
                inputdim=in_dim, 
                outdim=out_dim,
                gridsize=grid_size,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])
        
        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(in_dim) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])
        
        
    def forward(self, x, normalize=False):
        if normalize:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-6).detach() 
            x /= stdev
        
        # x = self.layer_norm[0](x)
        enc_x = self.layers[0](x)
        hid_x = enc_x
        
        for layer, layernorm in zip(self.layers[1:-1], self.layer_norm[1:-1]):
            # hid_x = layernorm(hid_x)
            hid_x = layer(hid_x)
            
        # hid_x = self.layer_norm[-1](hid_x)
        hid_x = self.layers[-1](hid_x)
        
        if normalize:
            hid_x = hid_x * stdev
            hid_x = hid_x + means
        return hid_x