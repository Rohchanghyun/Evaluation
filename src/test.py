import torch
import torch.nn as nn
import torchvision.models as models
from src import modules, mod, utils
import pdb
import transformers
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from transformers.models.clip.modeling_clip import CLIPTextTransformer,CLIPTextModel

class IntermediateLayerGetter(nn.Module):
    def __init__(self, model, return_layers):
        super(IntermediateLayerGetter, self).__init__()
        self.model = model
        self.return_layers = return_layers
        
    def forward(self, x):
        out = {}
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
            if len(out) == len(self.return_layers):
                break
        return out

def image_text_mapping(resnet, tokenizer, text_encoder, prompt ,input_ids, image, text_encoder_use_attention_mask=None):
    device = text_encoder.device 
   
    return_layers = {'layer2': 'layer2', 'layer3': 'layer3', 'layer4': 'layer4'}
    model = IntermediateLayerGetter(resnet, return_layers).to(device)
    
    global_max_pool = nn.AdaptiveMaxPool2d((1, 1)).to(device)

    # Linear Layer to convert each feature to [1, 384]
    linear_layer2 = nn.Linear(512, 384).to(device)
    linear_layer3 = nn.Linear(1024, 384).to(device)
    linear_layer4 = nn.Linear(2048, 384).to(device)

    input_image = image.to(device)
    outputs = model(input_image)

    pooled_outputs = {}
    for name, feature in outputs.items():
        pooled_feature = global_max_pool(feature).to(device)
        pooled_feature = pooled_feature.view(pooled_feature.size(0), -1)
        if name == 'layer2':
            pooled_feature = linear_layer2(pooled_feature)
        elif name == 'layer3':
            pooled_feature = linear_layer3(pooled_feature)
        elif name == 'layer4':
            pooled_feature = linear_layer4(pooled_feature)
        pooled_outputs[name] = pooled_feature
 
    concatenated_features = torch.cat(list(pooled_outputs.values()), dim=1).to(device)
    #for name, feature in pooled_outputs.items():
    #    print(f"{name} output feature shape after Global Max Pooling and Linear Layer: {feature.shape}")
  
    #print(f"Concatenated feature shape: {concatenated_features.shape}")
    text_encoder.text_model.forward = mod.forward_texttransformer.__get__(text_encoder.text_model, CLIPTextTransformer)
    
    text_encoder.forward = mod.forward_textmodel.__get__(text_encoder, CLIPTextModel)
    img2text = modules.IMG2TEXT(384*3,384*3,768).to(device)
    
    embeded_features_1, embeded_features_2 = img2text(concatenated_features)
    
    identifier='f'
    input_ids = input_ids.to(device)
    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(device)
    else:
        attention_mask = None
    ids = tokenizer(
                    prompt,
                    padding="do_not_pad",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                ).input_ids
    placeholder_token_id=tokenizer(
                    identifier,
                    padding="do_not_pad",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                ).input_ids[1]
    assert placeholder_token_id in ids,'identifier does not exist in prompt'
    pos_id = ids.index(placeholder_token_id)
    input_ids = tokenizer.pad(
        {"input_ids": [ids]},
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    input_ids = input_ids.to(device)
    
    hidden_states = utils.get_clip_hidden_states(input_ids,text_encoder).to(dtype=torch.float32)
    hidden_states[[0], [pos_id]]=embeded_features_1.to(device)
    hidden_states[[0], [pos_id+1]]=embeded_features_2.to(device)
    pos_eot = input_ids.to(dtype=torch.int, device=hidden_states.device).argmax(dim=-1)
    encoder_hidden_states = text_encoder(hidden_states=hidden_states, pos_eot=pos_eot)[0]
    encoder_hidden_states = torch.stack([encoder_hidden_states] * 3).squeeze(1)

    return encoder_hidden_states

