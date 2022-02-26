import torch
from tqdm import tqdm
from utils.common import im2tensor, tensor2im


def run_inversion(inputs, net, opts, return_intermediate_results=False):
    y_hat, latent, weights_deltas, codes = None, None, None, None

    if return_intermediate_results:
        results_batch = {idx: [] for idx in range(inputs.shape[0])}
        results_latent = {idx: [] for idx in range(inputs.shape[0])}
        results_deltas = {idx: [] for idx in range(inputs.shape[0])}
    else:
        results_batch, results_latent, results_deltas = None, None, None

    for iter in range(opts.n_iters_per_batch):
        y_hat, latent, weights_deltas, codes, _ = net.forward(inputs,
                                                              y_hat=y_hat,
                                                              codes=codes,
                                                              weights_deltas=weights_deltas,
                                                              return_latents=True,
                                                              resize=True,
                                                              randomize_noise=False,
                                                              return_weight_deltas_and_codes=True)

        if "cars" in opts.dataset_type:
            if opts.resize_outputs:
                y_hat = y_hat[:, :, 32:224, :]
            else:
                y_hat = y_hat[:, :, 64:448, :]

        if return_intermediate_results:
            store_intermediate_results(results_batch, results_latent, results_deltas, y_hat, latent, weights_deltas)
        
        # resize input to 256 before feeding into next iteration
        if "cars" in opts.dataset_type:
            y_hat = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
        else:
            y_hat = net.face_pool(y_hat)

    if return_intermediate_results:
        return results_batch, results_latent, results_deltas
    return y_hat, latent, weights_deltas, codes


def store_intermediate_results(results_batch, results_latent, results_deltas, y_hat, latent, weights_deltas):
    for idx in range(y_hat.shape[0]):
        results_batch[idx].append(y_hat[idx])
        results_latent[idx].append(latent[idx].cpu().numpy())
        results_deltas[idx].append([w[idx].cpu().numpy() if w is not None else None for w in weights_deltas])


def run_prediction(face_model, images, fine_encoding=False,  return_weights_deltas=False):
    assert isinstance(images, list)
    
    if fine_encoding:
        face_model.set_optimization_tools()
    
    results = {
        "image_original": [],
        "image": [],
        "latent": [],
        "style": [],
    }
    
    if return_weights_deltas:
        results.update({
            "weights_deltas": [],
        })
    
    for image in tqdm(images):
        # Convert to tensor after face alignment
        x_original, image_original = im2tensor(
            image, 
            align=True, 
            resize=(256, 256), 
            device=face_model.device, 
            return_original=True
        )

        # Encoding
        latent, weights_deltas = face_model.encoder(x_original, fine_encoding)
        style = face_model.latent2style(latent)

        # Decoding
        x = face_model.decoder(latent, weights_deltas)
        image = tensor2im(x[0])
        image_original = image_original.resize(image.size)
        
        # Results   
        results["image_original"].append(image_original)
        results["image"].append(image)
        results["latent"].append(latent)
        results["style"].append(style)
        
        if return_weights_deltas:
            results["weights_deltas"].append(weights_deltas)
    
    return results