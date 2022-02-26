# Wu et al., "Stylespace Analysis: Disentangled Controls For StyleGAN Image Generation", CVPR, 2021
# https://arxiv.org/pdf/2011.12799.pdf

edit_attributes = {
    "hair": {
        "level": "style",
        "index": (6, 364)
    },       
    "beard_goatee": {
        "level": "style",
        "index": (9, 421)
    },      
    "mouth_smiling": {
        "level": "style",
        "index": (6, 501)
    },
    "mouth_lipstick": {
        "level": "style",
        "index": (15, 45)
    },
    "eyes_makeup": {
        "level": "style",
        "index": (12, 414)
    },      
    "gaze": {
        "level": "style",
        "index": (9, 409)
    },
    "eyebrows": {
        "level": "style",
        "index": (8, 28)
    },    
    "gender": {
        "level": "style",
        "index": (9, 6)
    },   
    "pose": {
        "level": "latent",
        "direction": "pose"
    },
    "age": {
        "level": "latent",
        "direction": "age"
    },
}