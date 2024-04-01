import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from core.models import codi
from core.models.common.get_model import get_model
import torch
from core.models.ema import LitEma
from core.models.common.get_optimizer import get_optimizer

def load_yaml_config(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

class ConfigObject(object):
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


### モデルの定義=============================================

# AudioLDM
audioldm_cfg = load_yaml_config('configs/model/audioldm.yaml')
audioldm = ConfigObject(audioldm_cfg["audioldm_autoencoder"])

# Optimus
optimus_cfg = load_yaml_config('configs/model/optimus.yaml')

# optimus_vaeのconfigの辞書を、オブジェクトに置き換え
optimus_cfg['optimus_vae']['args']['encoder'] = ConfigObject(optimus_cfg['optimus_bert_encoder'])
optimus_cfg['optimus_vae']['args']['encoder'].args['config'] = ConfigObject(optimus_cfg['optimus_bert_encoder']['args']['config'])
optimus_cfg['optimus_vae']['args']['decoder'] = ConfigObject(optimus_cfg['optimus_gpt2_decoder'])
optimus_cfg['optimus_vae']['args']['decoder'].args['config'] = ConfigObject(optimus_cfg['optimus_gpt2_decoder']['args']['config'])
optimus_cfg['optimus_vae']['args']['tokenizer_encoder'] = ConfigObject(optimus_cfg['optimus_bert_tokenizer'])
optimus_cfg['optimus_vae']['args']['tokenizer_decoder'] = ConfigObject(optimus_cfg['optimus_gpt2_tokenizer'])
optimus_cfg['optimus_vae']['args']['args'] = ConfigObject(optimus_cfg['optimus_vae']['args']['args'])
optimus = ConfigObject(optimus_cfg["optimus_vae"])

# CLAP
clap_cfg = load_yaml_config('configs/model/clap.yaml')
clap = ConfigObject(clap_cfg["clap_audio"])

# CoDi
unet_cfg = load_yaml_config('configs/model/openai_unet.yaml')
unet_cfg["openai_unet_codi"]["args"]["unet_image_cfg"] = ConfigObject(unet_cfg["openai_unet_2d"])
unet_cfg["openai_unet_codi"]["args"]["unet_text_cfg"] = ConfigObject(unet_cfg["openai_unet_0dmd"])
unet_cfg["openai_unet_codi"]["args"]["unet_audio_cfg"] = ConfigObject(unet_cfg["openai_unet_2d_audio"])
unet = ConfigObject(unet_cfg["openai_unet_codi"])

# CoDiモデルのインスタンスを作成
model = codi.CoDi(audioldm_cfg=audioldm, optimus_cfg=optimus, clap_cfg=clap, unet_config=unet)

# データセット
dataset = MusicCapsDataset(csv_file='musiccaps_public.csv',
                           audio_dir='musiccaps_30')
                           
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


ema = LitEma(model)
optimizer_config = {
            'type': 'adam',
            'args': {
                 'weight_decay': 1e-4  # Weight decay
            }
        }
optimizer_config = ConfigObject(optimizer_config)
optimizer = get_optimizer()(model, optimizer_config)

### 学習ループ=============================================
for epoch in range(num_epochs=1):
    for batch_idx, (texts, audios) in enumerate(dataloader):
        # ここでモデルに入力を与え、損失を計算し、オプティマイザーを使用してモデルの重みを更新
        optimizer.zero_grad()
        loss = model.forward() #損失計算
        loss.backward()
        optimizer.step()

        # EMAの更新
        ema.update(model.parameters())

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')