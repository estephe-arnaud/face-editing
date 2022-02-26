import torch
from configs.paths_config import edit_paths


def edit_latent(latent, direction, factor):
    direction = {
        'age': torch.load(edit_paths['age']).cuda(),
        'smile': torch.load(edit_paths['smile']).cuda(),
        'pose': torch.load(edit_paths['pose']).cuda()
    }[direction]
    
    latent += factor * direction    
    return latent
    
def edit_style(style, index, delta):
    style_index = [0,1,1,2,2,3,4,4,5,6,6,7,8,8,9,10,10,11,12,12,13,14,14,15,16,16]
    i = style_index[index[0]]
    j = index[1]
    
    style[i][:, j] += delta
    return style