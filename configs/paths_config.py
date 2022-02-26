model_paths = {
	# models for backbones and losses
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'resnet34': 'pretrained_models/resnet34-333f7ec4.pth',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pt',
	# stylegan2 generators
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	# model for face alignment
	'shape_predictor': 'pretrained_models/shape_predictor_68_face_landmarks.dat',
	# models for ID similarity computation
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	# WEncoders for training on various domains
	'faces_w_encoder': 'pretrained_models/faces_w_encoder.pt',
	# models for domain adaptation
	'restyle_e4e_ffhq': 'pretrained_models/restyle_e4e_ffhq_encode.pt',
	'stylegan_pixar': 'pretrained_models/pixar.pt',
	'stylegan_toonify': 'pretrained_models/ffhq_cartoon_blended.pt',
	'stylegan_sketch': 'pretrained_models/sketch.pt',
	'stylegan_disney': 'pretrained_models/disney_princess.pt'
}

edit_paths = {
	'age': 'interfacegan_directions/age.pt',
	'smile': 'interfacegan_directions/smile.pt',
	'pose': 'interfacegan_directions/pose.pt',
}