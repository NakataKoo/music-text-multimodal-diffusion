my research

## Reference
The code structure is based on [Versatile Diffusion](https://github.com/SHI-Labs/Versatile-Diffusion). The audio diffusion model is based on [AudioLDM](https://github.com/haoheliu/AudioLDM). The video diffusion model is partially based on Make-A-Video.

![スクリーンショット 2024-02-24 203610](https://github.com/NakataKoo/music-text-multimodal-diffusion/assets/59306727/298197d2-82b6-4bc3-a3dc-80a83d1356e8)
[Versatile Diffusion](https://github.com/SHI-Labs/Versatile-Diffusion)

![スクリーンショット 2024-02-24 203507](https://github.com/NakataKoo/music-text-multimodal-diffusion/assets/59306727/bdfb42ae-cefb-49c9-b601-4d72a386410b)
[AudioLDM](https://github.com/haoheliu/AudioLDM)

## Future work

We intend to exclude all modules related to images and videos, and create a model dedicated to Any-to-Any generation of music and text. In doing so, we will also consider replacing the [AudioLDM](https://github.com/haoheliu/AudioLDM)-based audio module with a music module based on [MusicLDM](https://github.com/RetroCirce/MusicLDM/tree/main), etc.

It will also be necessary to retrain the model on a music-text pair dataset
