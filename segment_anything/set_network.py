import sys
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from segment_anything.build_sam3D import sam_model_registry3D


def freeze_control(args, model):
    # 加载整个预训练模型的权重
    pretrained_state_dict = torch.load(args.sam_ckpt, map_location='cpu')['model_state_dict']
    # 将整个预训练模型的权重加载到模型
    model.load_state_dict(pretrained_state_dict, strict=False)

    unfreeze_layers = [
        'image_encoder.pos_embed',
        *[f'image_encoder.blocks.{i}.adapter1' for i in range(len(model.image_encoder.blocks))],
        *[f'image_encoder.blocks.{i}.adapter2' for i in range(len(model.image_encoder.blocks))],
        *[f'mask_decoder.transformer.layers.{i}.adapter1' for i in range(len(model.mask_decoder.transformer.layers))],
        *[f'mask_decoder.transformer.layers.{i}.adapter2' for i in range(len(model.mask_decoder.transformer.layers))],
        *[f'mask_decoder.transformer.layers.{i}.adapter3' for i in range(len(model.mask_decoder.transformer.layers))],
        *[f'mask_decoder.output_hypernetworks_mlps.{i}.1]' for i in range(len(model.mask_decoder.output_hypernetworks_mlps))],
        'mask_decoder.adapter_mask_tokens',
        ]

    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in unfreeze_layers):
            param.requires_grad = True
            #print(name, 'is Unfreeze')
        else:
            param.requires_grad = False
            #print(name, 'is Freeze')
        
    return model