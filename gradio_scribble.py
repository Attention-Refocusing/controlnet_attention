from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from annotator.pidinet import PidiNetDetector
from annotator.util import nms
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import os
from annotator.oneformer import OneformerCOCODetector, OneformerADE20kDetector

preprocessor = None

model_name = 'control_v11p_sd15_scribble'
# model = create_model(f'/vulcanscratch/quynhpt/ControlNet-v1-1-nightly/models/{model_name}.yaml').cpu()
# model.load_state_dict(load_state_dict('/vulcanscratch/quynhpt/ControlNet-v1-1-nightly/models/v1-5-pruned.ckpt', location='cuda'), strict=False)
# model.load_state_dict(load_state_dict(f'/vulcanscratch/quynhpt/ControlNet-v1-1-nightly/models/{model_name}.pth', location='cuda'), strict=False)
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def generate_seg_paper(detected_map):
    gray_big = cv2.cvtColor(detected_map, cv2.COLOR_BGR2GRAY)
    # class1 = 151
    # class1 = 208
    # class2 = 57 # teddy
    # class2 =55
    class2 = 57  # boy
    class1 = 55 #bo
    bi_linear = np.zeros_like(gray_big)
    # import pdb; pdb.set_trace()
    bi_linear[np.where(gray_big==class1) ] = 1
    bi_linear[np.where(gray_big==class2) ] = 1
    bi3 = np.expand_dims(bi_linear, axis=-1)
    bi3 =  np.repeat(bi3, 3, axis=-1)
    seg_img = detected_map *bi3
    seg_img[seg_img==0] = 255
    cv2.imwrite('paper/boy_teddy.png',seg_img)
    assert False
    

def get_segment(det, input_image):
    global preprocessor
    if det == 'Seg_OFCOCO':
        if not isinstance(preprocessor, OneformerCOCODetector):
            preprocessor = OneformerCOCODetector()
    if det == 'Seg_OFADE20K':
        if not isinstance(preprocessor, OneformerADE20kDetector):
            preprocessor = OneformerADE20kDetector()
    if det == 'Seg_UFADE20K':
        if not isinstance(preprocessor, UniformerDetector):
            preprocessor = UniformerDetector()

    # import pdb; pdb.set_trace()
    class2 = 57  # boy
    class1 = 55 #boy
    # # class2=208 # dog
    # # class1 = 151
    # class1 = 208
    # class1 = 164
    # class2 = 112
    # class1 = 204
    # class2 = 81
    input_image = HWC3(input_image)
    

    if det == 'None':
        detected_map = input_image.copy()
    else:
        detected_map = preprocessor(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
    generate_seg_paper(detected_map)
    
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    W_attn, H_attn = int(W/32), int(H/32)
    print(W_attn, H_attn)
    #
    # detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    #resize mask
    detected_map = cv2.resize(detected_map, (W_attn, H_attn),interpolation=cv2.INTER_LINEAR)
    # import pdb; pdb.set_trace()
    gray = cv2.cvtColor(detected_map, cv2.COLOR_BGR2GRAY)
    # import pdb; pdb.set_trace()
    mask_1 = np.zeros((H_attn, W_attn))
    mask_1 [gray==class1] = 1.
    mask_2 = np.zeros((H_attn, W_attn))
    mask_2[gray==class2]=1.
    cv2.imwrite('scrib/boy/mask1.png',mask_1)
    cv2.imwrite('scrib/boy/mask2.png',mask_2)
    

def process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    global preprocessor

    if 'HED' in det:
        if not isinstance(preprocessor, HEDdetector):
            preprocessor = HEDdetector()

    if 'PIDI' in det:
        if not isinstance(preprocessor, PidiNetDetector):
            preprocessor = PidiNetDetector()

    # with torch.no_grad():
    if True:
        input_image = HWC3(input_image)

        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results

if __name__=='__main__':
    det = "Scribble_HED"
    # input_image = cv2.imread('/vulcanscratch/quynhpt/ControlNet-v1-1-nightly/img/cat_dog.png')
    # input_image = cv2.resize(input_image,(832,512))
    # img_names = os.listdir('try_img/img_sheep')
    # img_names = img_names[0:4]
    # for n in img_names:
    # print (n)
    input_image = cv2.imread('img/cat.jpeg')
    # import pdb; pdb.set_trace()
    # n = 
    # input_image = cv2.imread('/vulcanscratch/quynhpt/ControlNet-v1-1-nightly/try_img/img/boy_teddy.jpeg')
    # input_image = cv2.imread('/vulcanscratch/quynhpt/ControlNet-v1-1-nightly/img_depth/baby/baby_harder.jpeg')
    # input_image = cv2.imread('/vulcanscratch/quynhpt/ControlNet-v1-1-nightly/img/cat.jpeg')
    # input_image = cv2.imread('')
    prompt = 'a cat and a dog'
    # prompt = 'a cat and a dog sitting on the bed room'
    # prompt = 'a baby and a teddy bear sitting on the ground'
    # prompt = 'a dog and a sheep'
    a_prompt = ', cute, detail, realistic,high-quality,HD'
    n_prompt = 'low quality'
    num_samples = 1
    image_resolution = 512
    detect_resolution = 512
    ddim_steps = 20
    guess_mode = False
    strength = 1.0
    scale = 9.0
    is_safe = False
    
    eta = 1.0
    # det_seg = 'Seg_OFCOCO'
    # get_segment(det_seg, input_image)
    for i in range(0,5):
        # if (i!=2): continue
        seed = i
        results = process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
        folder_path = 'scrib_result'
        os.makedirs(folder_path, exist_ok=True)
        cv2.imwrite(os.path.join(folder_path, 'loss.png'),results[0])
        cv2.imwrite(os.path.join(folder_path, 'loss' + str(i)+'_.png'),results[1][:,:,::-1])
        # break



# block = gr.Blocks().queue()
# with block:
#     with gr.Row():
#         gr.Markdown("## Control Stable Diffusion with Synthesized Scribble")
#     with gr.Row():
#         with gr.Column():
#             input_image = gr.Image(source='upload', type="numpy")
#             prompt = gr.Textbox(label="Prompt")
#             run_button = gr.Button(label="Run")
#             num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
#             seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=12345)
#             det = gr.Radio(choices=["Scribble_HED", "Scribble_PIDI", "None"], type="value", value="Scribble_HED", label="Preprocessor")
#             with gr.Accordion("Advanced options", open=False):
#                 image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
#                 strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
#                 guess_mode = gr.Checkbox(label='Guess Mode', value=False)
#                 detect_resolution = gr.Slider(label="Preprocessor Resolution", minimum=128, maximum=1024, value=512, step=1)
#                 ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
#                 scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
#                 eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
#                 a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
#                 n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
#         with gr.Column():
#             result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
#     ips = [det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
#     run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


# block.launch(server_name='0.0.0.0')
