import os
import pandas as pd
from torch.utils.data import Dataset
import torchaudio

class MusicCapsDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transform=None):
        self.audio_dir = audio_dir
        self.transform = transform
        self.data = []
        
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
        audio_path = os.path.join(self.audio_dir, f"{row['ytid']}.wav")
        caption = row['caption']
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, caption
