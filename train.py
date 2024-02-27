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

# データセットの前処理を定義します（例）=============================================
def your_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        # 必要に応じて他の変換を追加
    ])

# データセットのインスタンスを作成
train_dataset = YourDataset(root='./data', transform=your_transforms())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


def load_yaml_config(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

class ConfigObject(object):
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


### データセットの前処理とロード
transform = transforms.Compose([
    transforms.ToTensor(),
    # 他の前処理ステップがあればここに追加
])



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


### 学習=============================================

# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
# optimizer = LitEma(model, lr=0.001, decay=0.9999)

# 学習ループ
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # フォワードパス
        outputs = model(data)
        loss = criterion(outputs, targets)

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # エポックの終了ごとにログを出力
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# モデルの評価
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
