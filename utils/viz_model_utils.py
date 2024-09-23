import torch
import torch.nn as nn
import re
from transformers import BertTokenizer, BertModel
from transformers.models.bert.configuration_bert import BertConfig
from torchvision import transforms
from PIL import Image
from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

class VLTransformerITM(nn.Module):
    def __init__(self, text_encoder=None, config_bert=''):
        super().__init__()
        bert_config = BertConfig.from_json_file(config_bert)
        self.visual_encoder = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        self.text_encoder = BertModel.from_pretrained(
            text_encoder, config=bert_config, add_pooling_layer=False
        )
        self.itm_head = nn.Linear(768, 2)

    def forward(self, image, text):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        output = self.text_encoder(
            text.input_ids, attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,
            return_dict=True
        )
        vl_embeddings = output.last_hidden_state[:, 0, :]
        vl_output = self.itm_head(vl_embeddings)
        return vl_output


def pre_caption(caption, max_words=30):
    caption = re.sub(r"([,.'!?\"()*#:;~])", '', caption.lower()).replace('-', ' ').replace('/', ' ')
    caption = re.sub(r"\s{2,}", ' ', caption).rstrip('\n').strip(' ')
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption


normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711)
)
transform = transforms.Compose([
    transforms.Resize((384, 384), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize
])
