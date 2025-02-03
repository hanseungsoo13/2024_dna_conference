import os
import logging
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
import torchvision
import torchvision.transforms as T

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import model.alpha_clip as alpha_clip
from model.clip import _transform, load
from model.model import convert_weights, IM2TEXT, FiLMedIM2TEXT, IM_TRANSFORMER
from params import parse_args_from_yaml
from utils import convert_models_to_fp32

def load_model_alphaclip(args):
    model, preprocess_val = alpha_clip.load("ViT-L/14", device=args.gpu, 
                                        alpha_vision_ckpt_pth="./checkpoints/clip_l14_grit+mim_fultune_6xe.pth", 
                                        lora_adapt=False, rank=-1)
    
    img2text = IM2TEXT(embed_dim=model.embed_dim, 
                       middle_dim=args.middle_dim, 
                       output_dim=model.token_embedding.weight.shape[1],
                       n_layer=args.n_layer)

    # img2text = FiLMedIM2TEXT(embed_dim=model.embed_dim, 
    #                         middle_dim=args.middle_dim, 
    #                         output_dim=model.token_embedding.weight.shape[1],
    #                         n_layer=args.n_layer) 

    # img2text = IM_TRANSFORMER(num_query_token=1,
    #                         cross_attention_freq=2,
    #                         embed_dim=model.token_embedding.weight.shape[1])

    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)

    if not torch.cuda.is_available():
        model.float()
        img2text.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        img2text.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)
        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, 
                device_ids=[args.gpu], 
                find_unused_parameters=model.has_extra)
            img2text = torch.nn.parallel.DistributedDataParallel(img2text, 
                device_ids=[args.gpu], find_unused_parameters=False)
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
            img2text = torch.nn.DataParallel(img2text, device_ids=args.multigpu)

    if args.resume == 'auto':
        checkpoint_list = os.listdir(args.checkpoint_path)
        checkpoint_list = [ckpt for ckpt in checkpoint_list if ckpt.startswith('epoch')]
        if checkpoint_list:
            latest_epoch = max([int(ckpt.split('_')[1].split('.')[0]) for ckpt in checkpoint_list])
            args.resume = os.path.join(args.checkpoint_path, f'epoch_{latest_epoch}.pt')
        else:
            args.resume = None

    assert args.resume is not None
    if os.path.isfile(args.resume):
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            loc = "cuda:{}".format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        sd = checkpoint["state_dict"]
        sd_img2text = checkpoint["state_dict_img2text"]
        if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        if not args.distributed and next(iter(sd_img2text.items()))[0].startswith('module'):
            sd_img2text = {k[len('module.'):]: v for k, v in sd_img2text.items()}
        model.load_state_dict(sd)
        img2text.load_state_dict(sd_img2text)
        logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']}')")
    else:
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
    return model, img2text, preprocess_val

def load_model_pic2word(args):
    model, preprocess_train, preprocess_val = load(
            args.model,
            jit=False)
    img2text = IM2TEXT(embed_dim=model.embed_dim, 
                           middle_dim=args.middle_dim, 
                           output_dim=model.token_embedding.weight.shape[1], 
                           n_layer=args.n_layer)

    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)

    if not torch.cuda.is_available():
        model.float()
        img2text.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        img2text.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)
        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, 
                device_ids=[args.gpu], 
                find_unused_parameters=model.has_extra)
            img2text = torch.nn.parallel.DistributedDataParallel(img2text, 
                device_ids=[args.gpu], find_unused_parameters=False)
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
            img2text = torch.nn.DataParallel(img2text, device_ids=args.multigpu)

    if args.resume == 'auto':
        checkpoint_list = os.listdir(args.checkpoint_path)
        checkpoint_list = [ckpt for ckpt in checkpoint_list if ckpt.startswith('epoch')]
        if checkpoint_list:
            latest_epoch = max([int(ckpt.split('_')[1].split('.')[0]) for ckpt in checkpoint_list])
            args.resume = os.path.join(args.checkpoint_path, f'epoch_{latest_epoch}.pt')
        else:
            args.resume = None

    assert args.resume is not None
    if os.path.isfile(args.resume):
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            loc = "cuda:{}".format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        sd = checkpoint["state_dict"]
        sd_img2text = checkpoint["state_dict_img2text"]
        if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        if not args.distributed and next(iter(sd_img2text.items()))[0].startswith('module'):
            sd_img2text = {k[len('module.'):]: v for k, v in sd_img2text.items()}
        model.load_state_dict(sd)
        img2text.load_state_dict(sd_img2text)
        logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']}')")
    else:
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
    return model, img2text, preprocess_val

def get_dino():
    model_id = "IDEA-Research/grounding-dino-tiny"
    dino_processor = AutoProcessor.from_pretrained(model_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id) # .to('cuda')
    return dino_model, dino_processor

def get_sam():
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge") # .to('cuda')
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    return sam_model, sam_processor

def dino_process(images, texts, dino_model, dino_processor):
    inputs = dino_processor(images=images, text=texts, return_tensors='pt') # .to('cuda')
    with torch.no_grad():
        outputs = dino_model(**inputs)
    
    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[images.size[::-1]]
    )
    return results[0]['boxes'].unsqueeze(0).cpu().numpy().tolist()

def sam_process(images, input_boxes, sam_model, sam_processor):
    inputs = sam_processor(images, input_boxes=input_boxes, return_tensors="pt") # .to('cuda')
    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )

    combined_mask = None
    for mask in masks[0]:
        new_mask = mask.float()
        if combined_mask is None:
            combined_mask = new_mask
        else:
            combined_mask = np.maximum(combined_mask, new_mask)
    
    return combined_mask

def cosine_sim(vec1, vec2):
    cosine_similarity = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
    return cosine_similarity.item()

def euclidean_dis(vec1, vec2):
    euclidean_distance = torch.dist(vec1, vec2)
    return euclidean_distance.item()

def plot_similarity_heatmap(similarity_matrix, class_list, metric_name="Cosine Similarity"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, 
                xticklabels=class_list,
                yticklabels=class_list,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                vmin=-0.1,  # 최소값을 -0.1로 설정
                vmax=0.1)   # 최대값을 0.1로 설정
    plt.title(f'{metric_name} Heatmap between Classes')
    plt.xlabel('Classes')
    plt.ylabel('Classes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'QFormer_30_similarity_heatmap_{metric_name.lower().replace(" ", "_")}.png')
    plt.close()

if __name__ == "__main__":
    config_path = "./configs/projection_module_alphaclip.yml"
    args = parse_args_from_yaml(config_path)
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    model, img2text, preprocess_val = load_model_alphaclip(args)

    print('Start Logging model.')

    dino_model, dino_processor = get_dino()
    sam_model, sam_processor = get_sam()

    mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(0.5, 0.26)
        ])

    class_list = ['cat', 'dog', 'elephant', 'wolf', 'human', 'potato', 'dress', 'laptop', 'tree']
    num_classes = len(class_list)
    
    # Initialize similarity matrices
    cosine_similarity_matrix = np.zeros((num_classes, num_classes))
    
    # Get class embeddings
    token_text = alpha_clip.tokenize([f'a photo of {i}.' for i in class_list])
    token_text = token_text.cuda(args.gpu, non_blocking=True)
    text_embedding = model.token_embedding(token_text)
    
    class_embedding_list = []
    for embedding_per_class in text_embedding:
        class_embedding = embedding_per_class[4]
        class_embedding_list.append(class_embedding)

    # Process each image and compute similarities
    image_embeddings = []
    for i, image_class in enumerate(class_list):
        images = Image.open(f"/home/work/gisub_conference/2024_dna_conference/test_images/{image_class}.jpg")
        texts = f"A {image_class}."

        input_boxes = dino_process(images, texts, dino_model, dino_processor)
        image_maskes = sam_process(images, input_boxes, sam_model, sam_processor)

        images = preprocess_val(images).to('cuda')
        image_maskes = preprocess_val.transforms[0](image_maskes)
        image_maskes = preprocess_val.transforms[1](image_maskes)
        image_maskes = np.array(image_maskes)

        binary_maskes = (image_maskes[:, :, 0] == 255)
        alphas = mask_transform((binary_maskes * 255).astype(np.uint8))

        images = images.cuda(args.gpu, non_blocking=True)
        alphas = alphas.cuda(args.gpu, non_blocking=True)
        images = images.unsqueeze(0)
        alphas = alphas.unsqueeze(0)

        image_features, _ = model.visual(images, alphas, return_attn=True)
        # image_features = model.encode_image(images)
        image_features_query = img2text(image_features)
        image_embedding = image_features_query[0]
        image_embeddings.append(image_embedding)

        # Compute similarities
        for j, class_embedding in enumerate(class_embedding_list):
            similarity = cosine_sim(image_embedding, class_embedding)
            # similarity = euclidean_dis(image_embedding, class_embedding)
            cosine_similarity_matrix[i, j] = similarity
            print(f'Image {image_class} : text {class_list[j]} = {similarity}')
        print('-' * 50)

    # Plot heatmaps
    plot_similarity_heatmap(cosine_similarity_matrix, class_list, "Cosine Similarity")