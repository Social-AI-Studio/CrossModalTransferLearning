from datasets import load_dataset
from trl import SFTTrainer
from PIL import Image
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration
from peft import LoraConfig,get_peft_model
from datasets import load_dataset
from trl import SFTConfig
from huggingface_hub import login
import numpy as np
import os
import cv2
from itertools import chain
torch.cuda.empty_cache()


dataset_path = "" # path to load annotated dataset
output_dir = "" # path to store model check point

label_dict = {"hatemm": {0: "non-hateful", 1:"hateful"},
        "mhc": {0: "non-offensive", 1:"offensive"}}
for key in label_dict:
    if(key in dataset_path):
        dataset_name = key
prompt = """Text: '{text}' \n Analyze the image and text to determine if the content is """ + label_dict[dataset_name][2] + """. Respond with '""" + label_dict[dataset_name][2] + """' or '""" + label_dict[dataset_name][0] + """'—just one word, no sentences."""

def check_image_content(image_path, variance_threshold=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image could not be loaded.")
    variance = np.var(image)
    if variance > variance_threshold:
        return True
    else:
        return False

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

def format_data(sample):
    if("frames" in sample and sample["frames"] != None):
        i = 0
        while not check_image_content(f"{sample['frames']}/{i}.png") and os.path.exists(f"{sample['frames']}/{i+1}.png"):
            i += 1
        sample["image"] = f"{sample['frames']}/{i}.png"
    if("video" not in sample):
        sample["video"] = sample["image"]
   
    user_content = {
        "role": "user",
        "content": [
            {
                "type": mode,
                 mode: sample[mode],
            },
            {
                "type": "text",
                "text": prompt.format(text=sample["text"]),
            }
        ],
    }

    assistant_content = {
            "role": "assistant",
            "content": [{"type": "text", "text": label_dict[base_dataset][int(sample["label"])] }],
        }
    return {"input_ids": [
        user_content,
        assistant_content,
    ],
    }

def collate_fn(examples):
    batch = {}
    for sample in examples:
        messages = sample["input_ids"]
        labels = ""
        image = None
        video = []
        for message in messages:
            role = message["role"]
            for content in message["content"]:
                if content["type"] == "text":
                    if role == "assistant":
                        labels += content["text"] + " "

                if content["type"] == mode and role == "user":
                    if(mode == "video"):
                        if("video" in content and content["video"] != None and content["video"][-3:] == "mp4"):
                            try:
                                container = av.open(content['video'])
                                total_frames = container.streams.video[0].frames
                                indices = np.arange(0, total_frames, total_frames / 16).astype(int)
                                video = read_video_pyav(container, indices)
                            except:
                                continue
                        else:
                            image = Image.open(content['video']).convert('RGB')
                            image = image.resize((224, 224))
                            augmentation_transforms = transforms.Compose([
                            transforms.RandomRotation(degrees=15),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                            transforms.RandomResizedCrop(size=image.size, scale=(0.8, 1.0)),
                            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                            ])
                            video = np.stack([augmentation_transforms(image) for _ in range(16)])
                    elif(mode == "image"):
                            image = Image.open(content['image']).convert('RGB')
                            image = image.resize((224, 224))
        if(len(video) == 0 and image == None):
            continue
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        if(mode == "video"):
            inputs = processor(videos=video, text=input_text,
                            padding="max_length",
                            truncation=True,       
                            max_length=max_length, 
                            return_tensors="pt").to(model.device)
        elif(mode == "image"):
            inputs = processor(images=image, text=input_text, 
                                padding="max_length",
                                truncation=True,       
                                max_length=max_length, 
                                return_tensors="pt").to(model.device)

        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        if(mode == "video"):
            video_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.video_token)]
            for video_token_id in video_tokens:
                labels[labels == video_token_id] = -100
        elif(mode == "image"):
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100

        inputs["labels"] = labels
        
        for key in inputs:
            if(key not in batch):
                batch[key] = [inputs[key].squeeze(0)]
            else:
                batch[key].append(inputs[key].squeeze(0))

    for key in batch:
        batch[key] = torch.stack(batch[key])
    return batch

dataset = load_dataset('csv', data_files=dataset_path, split="train").shuffle(seed=42)  
dataset = [format_data(sample) for sample in dataset if sample["label"] != None and sample["type"] != "test"]
print("The data size is " + str(len(dataset)), flush=True)

model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
processor = AutoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf",
    padding="max_length", 
    truncation=True,     
    max_length=512,      
    return_tensors="pt")
tokenizer = processor.tokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16  
) 
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,  
    return_dict=True,
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    quantization_config=bnb_config,
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
for param in model.parameters():
    param.requires_grad = False 
for name, param in model.named_parameters():
    if "lora" in name:  
        param.requires_grad = True 

args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    max_grad_norm=0.3,
    dataloader_pin_memory=False,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=True,
    report_to="tensorboard",
    dataset_kwargs={"skip_prepare_dataset": True},
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=collate_fn,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

trainer.train()

peft_model = trainer.model
peft_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)




