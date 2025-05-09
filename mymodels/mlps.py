from myutils.mi_plugin import BaseBRDF
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim


    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

class PositionalEncoding(nn.Module):
    def __init__(self, L):
        """ L: number of frequency bands """
        super(PositionalEncoding, self).__init__()
        self.L= L
        
    def forward(self, inputs):
        L = self.L
        encoded = [inputs]
        for l in range(L):
            encoded.append(torch.sin((2 ** l * math.pi) * inputs))
            encoded.append(torch.cos((2 ** l * math.pi) * inputs))
        return torch.cat(encoded, -1)
class SineLayer(nn.Module):
    ''' Siren layer '''
    
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True, 
                 is_first=False, 
                 omega_0=30, 
                 weight_norm=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # self.init_weights()

        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)

    def init_weights(self):
        if self.is_first:
            nn.init.uniform_(self.linear.weight, 
                             -1 / self.in_features * self.omega_0, 
                             1 / self.in_features * self.omega_0)
        else:
            nn.init.uniform_(self.linear.weight, 
                             -np.sqrt(3 / self.in_features), 
                             np.sqrt(3 / self.in_features))
        nn.init.zeros_(self.linear.bias)

    def forward(self, input):
        return torch.sin(self.linear(input))
class SmoothClamp_real(nn.Module):
    def __init__(self, min_val=0., max_val=1, alpha=5.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.alpha = alpha 

    def forward(self, x):
        lower = self.min_val + (x - self.min_val) * torch.sigmoid(self.alpha * (x - self.min_val))
        upper = self.max_val - (self.max_val - x) * torch.sigmoid(self.alpha * (self.max_val - x))
        y = torch.where(x < self.min_val, lower, x)
        y = torch.where(x > self.max_val, upper, y)
        return y
class SmoothClamp(nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0, alpha=5.0):
        super(SmoothClamp, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.alpha = alpha 

    def forward(self, x):
        lower = self.min_val + (x - self.min_val) * torch.sigmoid(self.alpha * (x - self.min_val))
        upper = self.max_val - (self.max_val - x) * torch.sigmoid(self.alpha * (self.max_val - x))
        return torch.min(torch.max(x, lower), upper)

class PosMLP(BaseBRDF):

    def __init__(self,
                 in_dims,
                 out_dims,
                 dims,
                 skip_connection=(),
                 weight_norm=True,
                 multires_view=0,
                 output_type='envmap',color_ch = 5):
        super().__init__()
        self.init_range = np.sqrt(3 / dims[0])

        dims = [in_dims] + dims + [out_dims]
        first_omega = 1
        hidden_omega = 1
        self.output_type = output_type

        self.embedview_fn = lambda x: x

        if multires_view > 0:
            # breakpoint()
            embedview_fn, input_ch = get_embedder(multires_view, input_dims=2) #x,y
            self.embedview_fn = embedview_fn
            # breakpoint()
            dims[0] += (input_ch - in_dims) + color_ch
        # breakpoint()
        self.num_layers = len(dims)
        self.skip_connection = skip_connection

        for l in range(0, self.num_layers - 1):

            if l + 1 in self.skip_connection:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            is_first = (l == 0) and (multires_view == 0)
            is_last = (l == (self.num_layers - 2))

            if not is_last:
                omega_0 = first_omega if is_first else hidden_omega
                lin = SineLayer(dims[l], out_dim, True, is_first, omega_0,
                                weight_norm)
            else:
                lin = nn.Linear(dims[l], out_dim)
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
                if weight_norm:
                    lin = nn.utils.weight_norm(lin)
                    if torch.isnan(lin.weight).any():
                        raise ValueError(f'nan value in lin{l}.weight')

            setattr(self, "lin" + str(l), lin)

            # self.last_active_fun = nn.Tanh()
            # self.last_active_fun = nn.Identity()
            self.last_active_fun = nn.Softplus()
            # self.last_active_fun = nn.ReLU()
        pass

    def img2points(self, img):
        # img 1,h*w,5
        if img.shape[0]> 512:
            h = w = img.shape[0]**0.5
        else:
            h = (img.shape[0]/2)**0.5
            if not h.is_integer():
                raise ValueError('width should be double of height')
            w = h*2
        x_coords, y_coords = torch.meshgrid(torch.arange(h), torch.arange(w),indexing='ij')

        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()

        points = torch.stack([x_coords, y_coords], dim=1).to(img.device)
        embed_points = self.embedview_fn(points)
        # if embed_points.shape[-1] >2:
        #     breakpoint()
        points_w_color = torch.cat([embed_points, img], dim=1)
        return points_w_color

    def forward(self, img):
        points = self.img2points(img)
        # breakpoint()
        x = points

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if hasattr(lin, 'linear') and torch.isnan(lin.linear.weight).any():
                raise ValueError(f'nan value in lin{l}.weight')
            elif hasattr(lin, 'weight') and torch.isnan(lin.weight).any():
                raise ValueError(f'nan value in lin{l}.weight')

            if l in self.skip_connection:
                x = torch.cat([x, points], -1)

            x = lin(x)

            if torch.isnan(x).any():
                raise ValueError(f'nan value in x in lin{l}')
        if self.output_type == 'envmap':
            x = nn.Softplus()(x) # make sure positive
        elif self.output_type == 'arm':
            x = 1.3 * nn.Tanh()(x) + img
            x = x.clamp(0,1).detach() + x - x.detach()
            
        elif self.output_type == 'armn':
            arm = x[..., 0:5]
            arm = 1.3 * nn.Tanh()(arm) + img[...,0:5]
            arm = arm.clamp(0,1).detach() + arm - arm.detach()
            
            normal = x[..., 5:8]
            normal = nn.Tanh()(normal+img[...,5:8])
            out = torch.cat([arm,normal],dim=-1)
            return out
        elif self.output_type == 'normal':
            x = x + img
            x = nn.Tanh()(x)
            x = F.normalize(x, p=2, dim=-1)
        else:
            raise ValueError('output_type should be envmap or arm or armn')
        return x