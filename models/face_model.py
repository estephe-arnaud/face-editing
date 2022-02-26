import sys

sys.path.append(".")
sys.path.append("..")

import torch
import torch.nn.functional as F
from criteria.lpips.lpips import LPIPS
from optimizers.ranger import Ranger
from utils.inference_utils import run_inversion
from utils.model_utils import load_model
from utils.common import *


class FaceModel(torch.nn.Module):
    
    def __init__(self, device="cuda"):
        super().__init__()
        net, opts = load_model(
            checkpoint_path="pretrained_models/hyperstyle_ffhq.pt", 
            update_opts={"w_encoder_checkpoint_path": "pretrained_models/faces_w_encoder.pt"},
            device=device
        )

        self.device = device
        self.opts = opts
        self.net = net
        
        for param in self.net.parameters():
            param.requires_grad_(False) 
            
            
    def set_optimization_tools(self):
        self.loss_l2 = F.mse_loss
        self.loss_lpips = LPIPS(net_type='squeeze').eval().to(self.device)    
   
   
    def encoder(self, x, fine_encoding=False):  
        assert list(x.shape)[-2:] == [256, 256]
        _, latent0, weights_deltas0, _ = run_inversion(x, self.net, self.opts)
            
        if not fine_encoding:
            return latent0, weights_deltas0
                
        else:
            y = x.clone().detach()
            eyes = extract_eyes(y)
            mouth = extract_mouth(y)
            
            self.delta_latent = torch.nn.Parameter(torch.zeros_like(latent0).to(self.device))
                     
            optimizer = Ranger(self.parameters(), lr=1)   
                        
            for _ in range(100):  
                latent = latent0 + self.delta_latent
                
                y_hat = self.decoder(
                    latent=latent, 
                    weights_deltas=weights_deltas0, 
                    resize=True
                )                
                
                eyes_hat = extract_eyes(y_hat)
                mouth_hat = extract_mouth(y_hat)

                loss  = 2 * self.loss_l2(y_hat, y)
                loss += 2 * self.loss_lpips(y_hat, y)
                loss += 4 * self.loss_l2(eyes_hat, eyes)
                loss += 0 * self.loss_lpips(eyes_hat, eyes)
                loss += 4 * self.loss_l2(mouth_hat, mouth)
                loss += 0 * self.loss_lpips(mouth_hat, mouth)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            return latent, weights_deltas0


    def decoder(self, latent, weights_deltas=None, style=None, resize=False):
        generator = self.net.decoder
        
        if not style:
            y_hat, _ = generator(
                [latent], 
                weights_deltas=weights_deltas,
                input_is_latent=True, 
                randomize_noise=False, 
                return_latents=False
            )
            
        else:
            total_convs = len(generator.convs) + len(generator.to_rgbs) + 2   # +2 for first conv and toRGB
            if weights_deltas is None:
                weights_deltas = [None] * total_convs
                            
            noise = [getattr(generator.noises, 'noise_{}'.format(i)) for i in range(generator.num_layers)]
            
            out = generator.input(latent)
            out = conv_warper(generator.conv1, out, style[0], noise[0], weights_delta=weights_deltas[0])
            skip = generator.to_rgb1(out, latent[:, 1], weights_delta=weights_deltas[1])

            i = 1
            weight_idx = 2
            for conv1, conv2, noise1, noise2, to_rgb in zip(
                generator.convs[::2], generator.convs[1::2], noise[1::2], noise[2::2], generator.to_rgbs
            ):
                out = conv_warper(conv1, out, style[i], noise=noise1, weights_delta=weights_deltas[weight_idx])
                out = conv_warper(conv2, out, style[i+1], noise=noise2, weights_delta=weights_deltas[weight_idx+1])
                skip = to_rgb(out, latent[:, i + 2], skip, weights_delta=weights_deltas[weight_idx+2])

                i += 2
                weight_idx += 3

            y_hat = skip            

        if resize:
            y_hat = self.net.face_pool(y_hat)
            
        return y_hat
        

    def latent2style(self, latent):
        generator = self.net.decoder
        noise = [getattr(generator.noises, 'noise_{}'.format(i)) for i in range(generator.num_layers)]
        
        style = []
        style.append(generator.conv1.conv.modulation(latent[:, 0]))

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            generator.convs[::2], generator.convs[1::2], noise[1::2], noise[2::2], generator.to_rgbs
        ):
            style.append(conv1.conv.modulation(latent[:, i]))
            style.append(conv2.conv.modulation(latent[:, i+1]))
            i += 2
            
        return style
    
    
def conv_warper(layer, input, style, noise, weights_delta):
    conv = layer.conv
    batch, in_channel, height, width = input.shape

    style = style.view(batch, 1, in_channel, 1, 1)
    if weights_delta is None:
        weight = conv.scale * conv.weight * style
    else:
        weight = conv.scale * (conv.weight * (1 + weights_delta) * style)    

    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
    )

    if conv.upsample:
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        
    out = layer.noise(out, noise=noise)
    out = layer.activate(out)
    
    return out