import torch
import torchaudio
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from core.models import codi
from core.models.ema import LitEma
from core.models.common.get_optimizer import get_optimizer
from argparse import ArgumentParser
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from core.models.common.get_model import get_model
import warnings
warnings.filterwarnings('ignore')
import torch.multiprocessing as mp


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

# Audio Plot========================================================
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

### Model Define===============================================================

def model_define(x, c):

    if x == "audio" and c == "text":
        # AudioLDM
        audioldm_cfg = load_yaml_config('configs/model/audioldm.yaml')
        audioldm = ConfigObject(audioldm_cfg["audioldm_autoencoder"])

        # CLIP
        clip_cfg = load_yaml_config('configs/model/clip.yaml')
        clip = ConfigObject(clip_cfg["clip_frozen"])

        # Unet
        unet_cfg = load_yaml_config('configs/model/openai_unet.yaml')
        unet_cfg["openai_unet_codi"]["args"]["unet_audio_cfg"] = ConfigObject(unet_cfg["openai_unet_2d_audio"])
        unet = ConfigObject(unet_cfg["openai_unet_codi"])

        # CoDi
        codi_cfg = load_yaml_config('configs/model/codi.yaml')
        codi_cfg["codi"]["args"]["audioldm_cfg"] = audioldm
        codi_cfg["codi"]["args"]["clip_cfg"] = clip
        codi_cfg["codi"]["args"]["unet_config"] = unet
        codi = ConfigObject(codi_cfg["codi"])

        model = get_model()(codi)
        return model

    elif x == "text" and c == "audio":
        # Optimus
        optimus_cfg = load_yaml_config('configs/model/optimus.yaml')
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

        # Unet
        unet_cfg = load_yaml_config('configs/model/openai_unet.yaml')
        unet_cfg["openai_unet_codi"]["args"]["unet_text_cfg"] = ConfigObject(unet_cfg["openai_unet_0dmd"])
        unet = ConfigObject(unet_cfg["openai_unet_codi"])

        # CoDi
        codi_cfg = load_yaml_config('configs/model/codi.yaml')
        codi_cfg["codi"]["args"]["optimus_cfg"] = optimus
        codi_cfg["codi"]["args"]["clap_cfg"] = clap
        codi_cfg["codi"]["args"]["unet_config"] = unet
        codi = ConfigObject(codi_cfg["codi"])

        model = get_model()(codi)
        return model

    # AutoKL
    #autokl_cfg = load_yaml_config('configs/model/sd.yaml')
    #autokl = ConfigObject(autokl_cfg["sd_autoencoder"])

# Dataset=============================================================
class MusicCaps(Dataset):
    def __init__(self, csv_file, audio_dir, model, x, c, transform=None):
        self.audio_dir = audio_dir
        self.transform = transform
        self.data = []
        self.model = model
        self.x = x
        self.c = c
        
        all_data = pd.read_csv(csv_file)
        
        # Checks for the existence of audio files and adds only those data that exist to the list
        for idx, row in all_data.iterrows():
            audio_path = os.path.join(self.audio_dir, f"{row['ytid']}.wav")
            if os.path.exists(audio_path):
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        caption = row['caption'] # raw text
        audio_path = os.path.join(self.audio_dir, f"{row['ytid']}.wav")    
        waveform = torchaudio.load(audio_path) # raw audio（Tensor）

        if self.x == "audio" and self.c == "text":
            mel_latent = self.model.module.audioldm_encode(waveform[0]).detach() # transform mel-spectrogram（Tensor） into latent space
            text_emb = self.model.module.clip_encode_text([caption]).detach()
            return mel_latent, text_emb # data, condition
        elif self.x == "text" and self.c == "audio":
            text_latent = self.model.module.optimus_encode([caption]).detach()
            audio_emb = self.model.module.clap_encode_audio(waveform[0]).detach()
            return text_latent, audio_emb # data, condition

### Training=============================================

def train():

    parser = ArgumentParser('DDP usage example')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
    args = parser.parse_args()

    args.is_master = args.local_rank == 0
    x = os.environ['XTYPE']
    c = os.environ['CTYPE']

    # init
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    print(args.local_rank)

    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("model difine")
    model = model_define(x, c)
    model = model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank])

    # Optimizer
    ema = LitEma(model)
    optimizer_config = {
                'type': 'adam',
                'args': {
                    'weight_decay': 1e-4  # Weight decay
                }
            }
    optimizer_config = ConfigObject(optimizer_config)
    optimizer = get_optimizer()(model, optimizer_config)
    
    print("data load")
    dataset = MusicCaps(csv_file='/raid/m236866/md-mt/datasets/musiccaps/musiccaps-public.csv',
                        audio_dir='/raid/m236866/md-mt/datasets/musiccaps/musiccaps_30', 
                        model=model, 
                        x=x, 
                        c=c)
    sampler = DistributedSampler(dataset, rank=args.local_rank)
    dataloader = DataLoader(dataset, 
                            batch_size=6, 
                            sampler=sampler, 
                            collate_fn=collate_fn, 
                            pin_memory=True, 
                            num_workers=os.cpu_count(), 
                            multiprocessing_context='spawn')                

    torch.backends.cudnn.benchmark = True

    print("train start")
    num_epochs=2
    running_loss = 0
    for epoch in range(num_epochs):
        model.train()
        dist.barrier()
        for batch_idx, (data, condition) in enumerate(dataloader):
            print("epoch", epoch, "batch", batch_idx)
            optimizer.zero_grad()
            data = data.to(args.local_rank)
            condition = condition.to(args.local_rank)
            loss = model.forward(x=data, c=condition)
            loss.backward()
            optimizer.step()
            # EMA update
            ema.update(model.parameters())

            running_loss += loss*data.size(0)
            print(batch_idx)

        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
    print(running_loss)
    dist.destroy_process_group()

if __name__ == "__main__":

    train()