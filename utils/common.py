from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.align_faces_parallel import align_face
from utils.model_utils import STYLESPACE_CHANNELS


def im2tensor(image, align=False, resize=None, return_original=False, device="cuda"):
    if align:
        try:
            image = align_face(image)
        except Exception as e:
            print(e)
        image_original = image.copy()
    if resize is not None:
        image = image.resize(resize)
    img = np.array(image)
    img = 2 * (img / 255.) - 1
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img)
    tensor = tensor.float().unsqueeze(0).to(device)
    if return_original:
        return tensor, image_original
    return tensor


def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def show_images(images, cols=1, titles=None, save_path=None, factor=3):
	n_images = len(images)
	if titles is None: 
		titles = [''] * n_images
	fig = plt.figure(figsize=(factor * (1 * (n_images / cols) + 3), factor * (1 * cols  + 1)))
	fig.set_facecolor("white")
	plt.subplots_adjust(wspace=0, hspace=0.1)
	for n, (image, title) in enumerate(zip(images, titles)):
		a = fig.add_subplot(cols, int(np.ceil(n_images/float(cols))), n + 1)
		a.set_axis_off()
		if image.ndim == 2:
			plt.gray()
		plt.imshow(image)
		a.set_title(title, y=-0.1)
	fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
	if save_path is not None:
		fig.savefig(save_path)
	plt.close()
	return fig


def show_tensors(tensors, **kwargs):
    """
    tensors (list): list of tensors, range in [-1, 1]
    """
    arrays = []
    for t in tensors:
        assert len(t.size()) <= 4
        if len(t.size()) == 4:
            t = t[0]
        im = tensor2im(t)
        arrays.append(np.array(im))

    fig = show_images(arrays, **kwargs)
    return fig

    
def style2vec(style):
    assert isinstance(style, list)
    return torch.cat(style, 1)


def vec2style(vec):
    assert vec.ndim == 2
    assert vec.shape[1] == np.sum(STYLESPACE_CHANNELS)
    style = []
    a = 0
    for n_channels in STYLESPACE_CHANNELS:
        b = a + n_channels
        style.append(vec[:, a:b])
        a = b
    return style


def latent2vec(latent):
    return torch.nn.Flatten()(latent)


def vec2latent(vec):
    assert vec.ndim == 2
    assert vec.shape[1] == 9216
    shape = (vec.shape[0], 18, 512)
    return torch.reshape(vec, shape)
