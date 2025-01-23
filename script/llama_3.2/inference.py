import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel, PeftConfig
import pandas as pd
from tqdm import tqdm
import  os, cv2
import numpy as np
import torch.nn.functional as F

checkpoint_path = "" # path to load model check point
dataset_name = "" # target video dataset name
output_path = "" # path to store prediction results

few_shot = 0
label_dict = {"hatemm": {0: "non-hateful", 1:"non-hateful", 2:"hateful"},
            "mhc": {0: "non-offensive", 1:"offensive", 2:"offensive"}}

FEW_SHOT_LST = {
    "hatemm": ['non_hate_video_1', 'hate_video_67', 'hate_video_131', 'non_hate_video_276', 'hate_video_226', 'non_hate_video_350', 'hate_video_307', 'non_hate_video_418'],
    'mhc': ['pQbEa24u-HM', 'bONmkrUFgNs', '0BqXuLbTxks', 'YLUQBKOOGHc', 'uAEdVB7e4UM', 'lUZDahwzuX4', 'o0AzXXiIRNw', 'PDIR3yEIS2M'],
}
few_shot_samples = pd.read_csv(f'../../dataset/{dataset_name}.csv', index_col=False)
few_shot_samples = few_shot_samples[few_shot_samples['id'].isin(FEW_SHOT_LST[dataset_name])][:few_shot]

base_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
base_model = MllamaForConditionalGeneration.from_pretrained(
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

@torch.no_grad()
def inference(row):
    if few_shot > 0:
        messages, images = zip(*[create_example(row) for _, row in few_shot_samples.iterrows()])
        messages = [item for sublist in messages for item in sublist]
        images = list(images)
    else:
        messages = []
        images = []
    if("frames" in row and row["frames"] != None):
        i = 0
        while not check_image_content(f"{row['frames']}/{i}.png") and os.path.exists(f"{row['frames']}/{i+1}.png"):
            i += 1
        row["image"] = f"{row['frames']}/{i}.png"
    image = Image.open(row['image']).convert('RGB')
    image = image.resize((560, 560))
    images.append(image)

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": row['image'],
                },
                {"type": "text", "text": f"Text: '{row['text']}' \n Analyze the image and text to determine if the content is "+label_dict[dataset_name][2]+". Respond with '"+label_dict[dataset_name][2]+"' or '"+label_dict[dataset_name][0]+"'—just one word, no sentences."}
            ]
        }
    )
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    if(image != None):
        inputs = processor(images, text=input_text, return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=input_text, return_tensors="pt").to(model.device)

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

