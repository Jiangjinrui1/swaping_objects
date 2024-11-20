import cv2
import os
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention


from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import tqdm


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


# config = OmegaConf.load('./configs/inference.yaml')
# model_ckpt =  config.pretrained_model
# model_config = config.config_file

# model = create_model(model_config ).cpu()
# model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
# model = model.cuda()
# ddim_sampler = DDIMSampler(model)
model = None
ddim_sampler = None

# 加载 YOLOv5 模型
model_yolo = YOLO('yolov5l.pt')
sam_checkpoint = r"D:\Google_Browser_download\sam_vit_h_4b8939.pth" # 替换为您的权重文件路径
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)


def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask


def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]


    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image, (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # ref aug 
    masked_ref_image_aug = masked_ref_image #aug_data(masked_ref_image) 

    # collage aug 
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask #aug_data_mask(masked_ref_image, ref_mask) 
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2])

    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    item = dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ) ) 
    return item


def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, guidance_scale = 5.0):
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][:,:,-1] * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()


    clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H,W = 512,512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 100 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
    return gen_image
def swap_objects_in_image(image_path, mask1_path, mask2_path, save_path):
    """
    使用 AnyDoor 在同一张图片中交换两个实体的位置。

    Args:
        image_path (str): 输入图像路径。
        mask1_path (str): 实体 1 的掩码路径。
        mask2_path (str): 实体 2 的掩码路径。
        save_path (str): 结果保存路径。
    """
    # 加载原始图像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 加载掩码
    mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE) > 128
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE) > 128
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)

    # 提取实体 1 和实体 2 的图像
    obj1 = cv2.bitwise_and(image, image, mask=mask1)
    obj2 = cv2.bitwise_and(image, image, mask=mask2)

    # 生成背景图像（去除实体 1 和 2）
    bg_mask = 1 - (mask1 + mask2)
    background = cv2.bitwise_and(image, image, mask=bg_mask.astype(np.uint8))

    # 将实体 1 替换到实体 2 的位置
    gen_image1 = inference_single_image(obj1, mask1, background.copy(), mask2)

    # 将实体 2 替换到实体 1 的位置
    gen_image2 = inference_single_image(obj2, mask2, background.copy(), mask1)

    # 合并生成的图像
    combined = np.where(mask1[:, :, None], gen_image2, background)
    combined = np.where(mask2[:, :, None], gen_image1, combined)

    # 保存最终结果
    result_image = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, result_image)
    print(f"结果已保存到 {save_path}")
def extract_two_similar_objects(image_path, output_dir):
    """
    提取具有相同标签的两个目标及其掩码。

    Args:
        image_path (str): 输入图像路径。
        sam_checkpoint (str): SAM 模型权重路径。
        output_dir (str): 输出目录，用于保存目标图像和掩码。

    Returns:
        None: 如果检测不到符合条件的目标，直接跳过。
    """
    # 加载 YOLOv5 模型
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found:", image_path)
        return

    # 推理检测
    results = model(image)

    # 加载 SAM 模型
    predictor.set_image(image)

    # 提取检测结果，按标签分组
    objects = {}
    for r in results:
        for box in r.boxes:
            label = int(box.cls[0])  # 类别
            confidence = box.conf[0]  # 置信度
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边界框坐标
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 中心点

            # 保存到按类别分组的字典
            if label not in objects:
                objects[label] = []
            objects[label].append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "center": (cx, cy)
            })

    # 检查是否存在至少两个目标具有相同标签
    for label, objs in objects.items():
        if len(objs) < 2:
            continue  # 如果目标数少于 2，跳过此标签

        # 选择两个目标
        obj1, obj2 = objs[:2]

        # 使用 SAM 生成掩码
        masks = []
        for obj in [obj1, obj2]:
            input_points = np.array([obj["center"]])
            input_labels = np.array([1])
            mask, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels)
            masks.append(mask[0])

        # 裁剪目标图像
        obj_images = []
        for i, obj in enumerate([obj1, obj2]):
            x1, y1, x2, y2 = obj["bbox"]
            cropped_image = image[y1:y2, x1:x2]
            cropped_mask = masks[i][y1:y2, x1:x2]

            # 保存图像和掩码
            cv2.imwrite(f"{output_dir}/object_{label}_{i}.png", cropped_image)
            cv2.imwrite(f"{output_dir}/mask_{label}_{i}.png", cropped_mask.astype(np.uint8) * 255)

        print(f"Saved two objects with label {label} in {output_dir}")
        return  # 只保存一组，退出函数

    print("No matching objects found with at least two instances having the same label.")

def extract_and_swap_objects(image_path,save_path):
    """
    提取具有相同标签的两个目标及其掩码，并交换位置。

    Args:
        image_path (str): 输入图像路径。
        save_path (str): 输出图像保存路径。

    Returns:
        None: 如果没有符合条件的目标，直接跳过。
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found:", image_path)
        return

    # 推理检测
    results = model_yolo(image)

    # 加载 SAM 模型
    predictor.set_image(image)

    # 提取检测结果，按标签分组
    objects = {}
    for r in results:
        for box in r.boxes:
            label = int(box.cls[0])  # 类别
            confidence = box.conf[0]  # 置信度
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边界框坐标
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 中心点

            # 保存到按类别分组的字典
            if label not in objects:
                objects[label] = []
            objects[label].append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "center": (cx, cy)
            })

    # 检查是否存在至少两个目标具有相同标签
    for label, objs in objects.items():
        if len(objs) < 2:
            continue  # 如果目标数少于 2，跳过此标签

        # 选择两个目标
        obj1, obj2 = objs[:2]

        # 使用 SAM 生成掩码
        masks = []
        for obj in [obj1, obj2]:
            input_points = np.array([obj["center"]])
            input_labels = np.array([1])
            mask, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels)

            masks.append(mask[2])
        # 使用SAM 进行目标分割
        # 裁剪目标区域
        obj_images = []
        for i, obj in enumerate([obj1, obj2]):
            x1, y1, x2, y2 = obj["bbox"]
            cropped_image = image[y1:y2, x1:x2]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_mask = masks[i][y1:y2, x1:x2]
            cropped_mask = cropped_mask.astype(np.uint8)
            obj_images.append((cropped_image, cropped_mask, obj["bbox"]))

        # 获取背景图像（去除目标区域）
        bg_mask = 1 - (masks[0] + masks[1])
        # background = cv2.bitwise_and(image, image, mask=bg_mask.astype(np.uint8))
        background = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # 调用 AnyDoor 交换位置
        gen_image1 = inference_single_image(obj_images[0][0], obj_images[0][1], background.copy(), masks[1])
        # background = cv2.cvtColor(gen_image1.astype(np.uint8), cv2.COLOR_RGB2BGR)
        background = gen_image1.astype(np.uint8)
        gen_image2 = inference_single_image(obj_images[1][0], obj_images[1][1], background.copy(), masks[0])

        # 保存结果
        result_image = cv2.cvtColor(gen_image2, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_image)
        print(f"Saved swapped image to {save_path}")
        return  # 只处理第一组，退出函数

    print("No matching objects found with at least two instances having the same label.")
def batch_process_images(input_dir, output_dir, sam_checkpoint):
    """
    批量处理目录中的图像，提取两个具有相同类别的目标并交换位置。

    Args:
        input_dir (str): 输入图像目录路径。
        output_dir (str): 输出图像目录路径。
        sam_checkpoint (str): SAM 模型权重路径。

    Returns:
        None
    """
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取目录中的所有图像文件
    image_files = [
        os.path.join(input_dir, file) for file in os.listdir(input_dir)
        if file.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # 遍历所有图像文件并处理
    for image_path in tqdm(image_files, desc="Processing Images"):
        try:
            # 构造输出路径
            image_name = os.path.basename(image_path)
            save_path = os.path.join(output_dir, f"{image_name}" + ".png")

            # 调用单张图像的处理函数
            extract_and_swap_objects(image_path, save_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == '__main__': 
    
    # ==== Example for inferring a single image ===
    save_path = './output'
    image_path =r"D:\研究生阶段\研0\VSCode_workspace\MORE\data\data\MORE\img_org\total\0ce97a65-feb3-52ce-b4e1-dac18cb90a9f.jpg"
    extract_and_swap_objects(image_path,save_path)


    # reference_image_path = './examples/TestDreamBooth/FG/01.png'
    # bg_image_path = './examples/TestDreamBooth/BG/000000309203_GT.png'
    # bg_mask_path = './examples/TestDreamBooth/BG/000000309203_mask.png'
    # save_path = './examples/TestDreamBooth/GEN/gen_res.png'

    # # reference image + reference mask
    # # You could use the demo of SAM to extract RGB-A image with masks
    # # https://segment-anything.com/demo
    # image = cv2.imread( reference_image_path, cv2.IMREAD_UNCHANGED)
    # mask = (image[:,:,-1] > 128).astype(np.uint8)
    # # 可视化mask，展示图片
    # cv2.imshow('mask', mask)
    # image = image[:,:,:-1]
    # image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    # ref_image = image 
    # ref_mask = mask

    # # background image
    # back_image = cv2.imread(bg_image_path).astype(np.uint8)
    # back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

    # # background mask 
    # tar_mask = cv2.imread(bg_mask_path)[:,:,0] > 128
    # tar_mask = tar_mask.astype(np.uint8)
    
    # gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)
    # h,w = back_image.shape[0], back_image.shape[0]
    # ref_image = cv2.resize(ref_image, (w,h))
    # vis_image = cv2.hconcat([ref_image, back_image, gen_image])
    
    # cv2.imwrite(save_path, vis_image [:,:,::-1])
    
#     #'''
#     # ==== Example for inferring VITON-HD Test dataset ===

#     from omegaconf import OmegaConf
#     import os 
#     DConf = OmegaConf.load('./configs/datasets.yaml')
#     save_dir = './VITONGEN'
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)

#     test_dir = DConf.Test.VitonHDTest.image_dir
#     image_names = os.listdir(test_dir)
    
#     for image_name in image_names:
#         ref_image_path = os.path.join(test_dir, image_name)
#         tar_image_path = ref_image_path.replace('/cloth/', '/image/')
#         ref_mask_path = ref_image_path.replace('/cloth/','/cloth-mask/')
#         tar_mask_path = ref_image_path.replace('/cloth/', '/image-parse-v3/').replace('.jpg','.png')

#         ref_image = cv2.imread(ref_image_path)
#         ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

#         gt_image = cv2.imread(tar_image_path)
#         gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

#         ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:,:,0]

#         tar_mask = Image.open(tar_mask_path ).convert('P')
#         tar_mask= np.array(tar_mask)
#         tar_mask = tar_mask == 5

#         gen_image = inference_single_image(ref_image, ref_mask, gt_image.copy(), tar_mask)
#         gen_path = os.path.join(save_dir, image_name)

#         vis_image = cv2.hconcat([ref_image, gt_image, gen_image])
#         cv2.imwrite(gen_path, vis_image[:,:,::-1])
#     #'''

    

