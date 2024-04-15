"""
Load model from checkpoint.

For model inference:
The outputs are stored in an array as [number of output modalities, number of samples]
If I generate 4 samples of image + caption, the shape would be [2, 4]
"""

import os
from core.models.model_module_infer import model_module

model_load_paths = ['CoDi_encoders.pth', 'CoDi_text_diffuser.pth', 'CoDi_audio_diffuser_m.pth', 'CoDi_video_diffuser_8frames.pth']
inference_tester = model_module(data_dir='checkpoints/', pth=model_load_paths, fp16=False) # turn on fp16=True if loading fp16 weights
inference_tester = inference_tester.cuda()
inference_tester = inference_tester.eval()

# Give a prompt
prompt = "A beautiful oil painting of a birch tree standing in a spring meadow with pink flowers, a distant mountain towers over the field in the distance. Artwork by Alena Aenami"

# Generate image
images = inference_tester.inference(
                xtype = ['image'],
                condition = [prompt],
                condition_types = ['text'],
                n_samples = 1, 
                image_size = 256,
                ddim_steps = 50)
print(images[0][0])