import torch
import torch.nn as nn
import torchvision.models as models
import modules, mod, utils
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

# ResNet50 모델 불러오기
resnet50 = models.resnet50(pretrained=True)

# 필요한 레이어 이름과 반환될 이름 매핑
return_layers = {'layer2': 'layer2', 'layer3': 'layer3', 'layer4': 'layer4'}

# IntermediateLayerGetter를 사용하여 필요한 레이어의 출력을 얻기
model = IntermediateLayerGetter(resnet50, return_layers)

# Global Max Pooling Layer
global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

# Linear Layer to convert each feature to [1, 384]
linear_layer2 = nn.Linear(512, 384)
linear_layer3 = nn.Linear(1024, 384)
linear_layer4 = nn.Linear(2048, 384)

# 입력 이미지 크기
input_image = torch.randn(1, 3, 384, 192)

# 모델 실행
outputs = model(input_image)

# 각 레이어의 출력 크기를 Global Max Pooling으로 변환하고 [1, 384]로 변환
pooled_outputs = {}
for name, feature in outputs.items():
    pooled_feature = global_max_pool(feature)
    pooled_feature = pooled_feature.view(pooled_feature.size(0), -1)
    if name == 'layer2':
        pooled_feature = linear_layer2(pooled_feature)
    elif name == 'layer3':
        pooled_feature = linear_layer3(pooled_feature)
    elif name == 'layer4':
        pooled_feature = linear_layer4(pooled_feature)
    pooled_outputs[name] = pooled_feature

# 변환된 feature들을 연결
concatenated_features = torch.cat(list(pooled_outputs.values()), dim=1)

# 출력 크기 확인
for name, feature in pooled_outputs.items():
    print(f"{name} output feature shape after Global Max Pooling and Linear Layer: {feature.shape}")

print(f"Concatenated feature shape: {concatenated_features.shape}")

img2text = modules.IMG2TEXT(384*3,384*3,768)

embeded_features_1, embeded_features_2 = img2text(concatenated_features)

print(f"Concatenated feature shape: {embeded_features_1.shape}")
print(f"Concatenated feature shape: {embeded_features_2.shape}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16,safety_checker=None)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

#build f2d pipeline
pipe.text_encoder.text_model.forward = mod.forward_texttransformer.__get__(pipe.text_encoder.text_model, CLIPTextTransformer)
pipe.text_encoder.forward = mod.forward_textmodel.__get__(pipe.text_encoder, CLIPTextModel)

prompt = 'A photo of a f l'
identifier='f'

ids = pipe.tokenizer(
                prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=pipe.tokenizer.model_max_length,
            ).input_ids
placeholder_token_id=pipe.tokenizer(
                identifier,
                padding="do_not_pad",
                truncation=True,
                max_length=pipe.tokenizer.model_max_length,
            ).input_ids[1]
assert placeholder_token_id in ids,'identifier does not exist in prompt'
pos_id = ids.index(placeholder_token_id)


input_ids = pipe.tokenizer.pad(
        {"input_ids": [ids]},
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids

hidden_states = utils.get_clip_hidden_states(input_ids,pipe.text_encoder)
print(hidden_states.shape)
print(hidden_states)
print(pos_id)
hidden_states[[0], [pos_id]]=embeded_features_1.to(dtype=torch.float16)
hidden_states[[0], [pos_id+1]]=embeded_features_2.to(dtype=torch.float16)
pos_eot = input_ids.to(dtype=torch.int, device=hidden_states.device).argmax(dim=-1)

print(f"pos_eot shape: {pos_eot.shape}")

#text encoding
encoder_hidden_states = pipe.text_encoder(hidden_states=hidden_states, pos_eot=pos_eot)[0]



def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(text_input_ids)
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds