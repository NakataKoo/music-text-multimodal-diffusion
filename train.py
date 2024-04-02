import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchaudio
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from core.models import codi
from core.models.common.get_model import get_model
from core.models.ema import LitEma
from core.models.common.get_optimizer import get_optimizer
from torchaudio.transforms import MelSpectrogram
from torch.multiprocessing import spawn

def load_yaml_config(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

class ConfigObject(object):
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])

def collate_fn(batch):
    texts, audios = zip(*batch)
    # 最大のオーディオ長を見つける
    max_length = max(audio.shape[1] for audio in audios)
    # パディング
    audios_padded = torch.stack([torch.nn.functional.pad(audio, (0, max_length - audio.shape[1])) for audio in audios])
    texts = torch.stack(texts)
    return texts, audios_padded

# 音声可視化========================================================
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    # パワースペクトルをデシベル単位に変換
    spec_db = 10 * np.log10(spec + np.finfo(float).eps)
    
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(spec_db, origin='lower', aspect=aspect, cmap='viridis')
    
    if xmax:
        axs.set_xlim((0, xmax))
    
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


sample_rate = 48000

# メルスペクトルグラムの取得========================================================
n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128

mel_spectrogram = MelSpectrogram(
    sample_rate= 48000,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm='slaney',
    onesided=True,
    n_mels=n_mels,
)

### モデルの定義===============================================================

def model_define():
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

    # CLIP
    clip_cfg = load_yaml_config('configs/model/clip.yaml')
    clip = ConfigObject(clip_cfg["clip_frozen"])

    # CoDiモデルのインスタンスを作成
    model = codi.CoDi(audioldm_cfg=audioldm, optimus_cfg=optimus, clip_cfg=clip, clap_cfg=clap, unet_config=unet)

    return model

# データセットの定義=============================================================
class MusicCapsTTM(Dataset):
    def __init__(self, csv_file, audio_dir, model, transform=None):
        self.audio_dir = audio_dir
        self.transform = transform
        self.data = []
        self.model = model
        
        # CSVファイルを読み込む
        all_data = pd.read_csv(csv_file)
        
        # 音声ファイルが存在するかどうかを確認し、存在するデータのみをリストに追加
        for idx, row in all_data.iterrows():
            audio_path = os.path.join(self.audio_dir, f"{row['ytid']}.wav")
            if os.path.exists(audio_path):
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        caption = row['caption'] # 生テキスト
        text_emb = self.model.module.clip_encode_text([caption])

        audio_path = os.path.join(self.audio_dir, f"{row['ytid']}.wav")    
        waveform = torchaudio.load(audio_path) # 生波形データ（Tensor）
        mel_latent = self.model.module.audioldm_encode(waveform[0]) # メルスペクトログラム（Tensor）の潜在表現に変換

        if self.transform:
            pass

        return text_emb, mel_latent 

### 学習ループ=============================================


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    torch.cuda.set_device(rank)
    setup(rank, world_size)
    
    # モデルを定義
    model = model_define()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Optimizerの定義
    ema = LitEma(model)
    optimizer_config = {
                'type': 'adam',
                'args': {
                    'weight_decay': 1e-4  # Weight decay
                }
            }
    optimizer_config = ConfigObject(optimizer_config)
    optimizer = get_optimizer()(model, optimizer_config)
    
    # データセット
    dataset = MusicCapsTTM(csv_file='/raid/m236866/md-mt/datasets/musiccaps/musiccaps-public.csv',
                            audio_dir='/raid/m236866/md-mt/datasets/musiccaps/musiccaps_30', model=model)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=5, sampler=sampler, collate_fn=collate_fn)                        

    # トレーニングループ
    num_epochs=2
    for epoch in range(num_epochs):
        for batch_idx, (texts, audios) in enumerate(dataloader):
            # ここでモデルに入力を与え、損失を計算し、オプティマイザーを使用してモデルの重みを更新
            optimizer.zero_grad()
            loss = model.forward(x=audios, c=texts) #損失計算
            loss.backward()
            optimizer.step()
            # EMAの更新
            ema.update(model.parameters())

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

    cleanup()

def main():
    # 使用するGPUの数
    world_size = 3
    
    # トレーニングプロセスの起動
    spawn(train,
          args=(world_size,),
          nprocs=world_size,
          join=True)

if __name__ == "__main__":
    main()