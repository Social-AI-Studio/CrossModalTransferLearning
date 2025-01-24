import torch
from PIL import Image
from transformers import LlavaNextVideoForConditionalGeneration, AutoProcessor, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel, PeftConfig
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import os, cv2
import numpy as np
import av
from torchvision import transforms
import torch.nn.functional as F
import csv

checkpoint_path = "" # path to load model check point
dataset_name = "" # target video dataset name
output_path = "" # path to store prediction results

mode = "video"
few_shot = 0
label_dict = {"hatemm": {0: "non-hateful", 1:"hateful"},
        "mhc": {0: "non-offensive", 1:"offensive"}}

FEW_SHOT_LST = {
    "hatemm": ['non_hate_video_1', 'hate_video_67', 'hate_video_131', 'non_hate_video_276', 'hate_video_226', 'non_hate_video_350', 'hate_video_307', 'non_hate_video_418'],
    'mhc': ['pQbEa24u-HM', 'bONmkrUFgNs', '0BqXuLbTxks', 'YLUQBKOOGHc', 'uAEdVB7e4UM', 'lUZDahwzuX4', 'o0AzXXiIRNw', 'PDIR3yEIS2M'],
}
description_path = "../../dataset/{dataset_name}_demo_description.csv'"
few_shot_samples = pd.read_csv(f'../../dataset/{dataset_name}.csv', index_col=False)
few_shot_samples = few_shot_samples[few_shot_samples['id'].isin(FEW_SHOT_LST[dataset_name])][:few_shot]

base_model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
base_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    base_model_id,    torch_dtype=torch.bfloat16,
).to("cuda:0")
base_model.generation_config.temperature=None
base_model.generation_config.top_p=None
processor = AutoProcessor.from_pretrained(base_model_id)

if(os.path.exists(checkpoint_path)):
    config = PeftConfig.from_pretrained(checkpoint_path)
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
else:
    print(checkpoint_path + " not exist.")
    model = base_model

def check_image_content(image_path, variance_threshold=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image could not be loaded.")
    variance = np.var(image)
    if variance > variance_threshold:
        return True
    else:
        return False

def create_example(row):
    if("frames" in row and row["frames"] != None):
        i = 0
        while not check_image_content(f"{row['frames']}/{i}.png") and os.path.exists(f"{row['frames']}/{i+1}.png"):
            i += 1
        row["image"] = f"{row['frames']}/{i}.png"

    image = Image.open(row['image']).convert('RGB')
    image = image.resize((560, 560))
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": row['image'],
                },
                {"type": "text", "text": f"Text: '{row['text']}' \n Analyze the image and text to determine if the content is "+label_dict[dataset_name][2]+". Respond with '"+label_dict[dataset_name][2]+"' or '"+label_dict[dataset_name][0]+"'—just one word, no sentences."}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": label_dict[dataset_name][int(row["label"])]}
            ]
        }

    ]
    return message, image

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frame_ndarray = frame.to_ndarray(format="rgb24")
            resized_frame = np.array(Image.fromarray(frame_ndarray).resize((224, 224)))
            frames.append(resized_frame)
    return np.stack(frames)

def read_csv_as_dict(file_path):
    data = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data[row['id']] = [row['label'], row['description'], row["text"]]
    return data

def create_demo():
    if(few_shot == 0):
        return ""
    demo = ""
    data = read_csv_as_dict(description_path)
    for i in range(few_shot):
        id = FEW_SHOT_LST[dataset_name][i]
        label, description, text = data[id]
        demo += (
            f"Video with vision content: '{description}' and Text: '{text}'. \n "
            f"Analyze the {mode} and text to determine if the content is {label_dict[dataset_name][2]}. "
            f"Respond with '{label_dict[dataset_name][2]}' or '{label_dict[dataset_name][0]}'—just one word, no sentences. \n "
            f"{label} \n "
        )
    demo += " Video with vision content as shown and "
    return demo 

@torch.no_grad()
def inference(row):
    if("frames" in row and row["frames"] != None):
        i = 0
        while not check_image_content(f"{row['frames']}/{i}.png") and os.path.exists(f"{row['frames']}/{i+1}.png"):
            i += 1
        row["image"] = f"{row['frames']}/{i}.png"
    if("video" not in row):
        row["video"] = row["image"]

    if(mode == "video"):
        if("video" in row and row["video"] != None and row["video"][-3:] == "mp4"):
            container = av.open(row['video'])
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / 16).astype(int)
            video = read_video_pyav(container, indices)
        else:
            image = Image.open(row['image']).convert('RGB')
            image = image.resize((224, 224))
            augmentation_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(size=image.size, scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            ])

            video = [augmentation_transforms(image) for _ in range(16)]
    elif(mode == "image"):
        image = Image.open(row['image']).convert('RGB')
        image = image.resize((224, 224))
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": mode,
                    mode: row[mode],
                },
                {"type": "text", "text": create_demo() + f"Text: '{row['text']}' \n Analyze the " + mode + " and text to determine if the content is "+label_dict[dataset_name][2]+". Respond with '"+label_dict[dataset_name][2]+"' or '"+label_dict[dataset_name][0]+"'—just one word, no sentences."}
            ]
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    if(mode == "video"):
        inputs = processor(videos=video, text=input_text, return_tensors="pt").to(model.device)
    elif(mode == "image"):
        inputs = processor(images=image, text=input_text, return_tensors="pt").to(model.device)
  
    prompt_length = inputs.input_ids.shape[-1]
    output = model.generate(**inputs, max_new_tokens=25, do_sample=False, return_dict_in_generate=True,output_logits=True)
    output = output.sequences

    if(os.path.exists(checkpoint_path)):
        response = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
    else:
        response = processor.decode(output[0][prompt_length:], skip_special_tokens=True)
    row['label'] = label_dict[dataset_name][int(row['label'])]
    return {'id': row['id'], 'response': response, 'label': row['label'] }

full_response = []
inference_data = load_dataset('csv', data_files=f'../../dataset/{dataset_name}.csv')
inference_data = inference_data.filter(lambda data: data['type'] == 'test')
for data in tqdm(inference_data['train']):
    if(data["label"] == None ):
        continue
    result = inference(data)
    full_response.append(result)
df = pd.DataFrame(full_response)
df.to_csv(output_path, index=False)
print('Done!')

