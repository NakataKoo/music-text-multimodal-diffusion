my research

## CoDi

We present Composable Diffusion (CoDi), a novel generative model capable of generating any combination of output modalities, such as language, image, video, or audio, from any combination of input modalities. Unlike existing generative AI systems, CoDi can generate multiple modalities in parallel and its input is not limited to a subset of modalities like text or image. Despite the absence of training datasets for many combinations of modalities, we propose to align modalities in both the input and output space. This allows CoDi to freely condition on any input combination and generate any group of modalities, even if they are not present in the training data. CoDi employs a novel composable generation strategy which involves building a shared multimodal space by bridging alignment in the diffusion process, enabling the synchronized generation of intertwined modalities, such as temporally aligned video and audio. Highly customizable and flexible, CoDi achieves strong joint-modality generation quality, and outperforms or is on par with the unimodal state-of-the-art for single-modality synthesis.

## Reference
The code structure is based on [Versatile Diffusion](https://github.com/SHI-Labs/Versatile-Diffusion). The audio diffusion model is based on [AudioLDM](https://github.com/haoheliu/AudioLDM). The video diffusion model is partially based on Make-A-Video.

## Future work

We intend to exclude all modules related to images and videos, and create a model dedicated to Any-to-Any generation of music and text. In doing so, we will also consider replacing the [AudioLDM](https://github.com/haoheliu/AudioLDM)-based audio module with a music module based on [MusicLDM](https://github.com/RetroCirce/MusicLDM/tree/main), etc.

It will also be necessary to retrain the model on a music-text pair dataset
