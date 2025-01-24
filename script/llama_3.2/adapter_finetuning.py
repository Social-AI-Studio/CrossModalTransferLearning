from datasets import load_dataset
from trl import SFTTrainer
from PIL import Image
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, MllamaForConditionalGeneration
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

def format_data(sample, prompt, dataset_name):
    if("frames" in sample and sample["frames"] != None):
        i = 0
        while not check_image_content(f"{sample['frames']}/{i}.png") and os.path.exists(f"{sample['frames']}/{i+1}.png"):
            i += 1
        sample["image"] = f"{sample['frames']}/{i}.png"

    user_content = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": sample["image"],
            },
            {
                "type": "text",
                "text": prompt.format(text=sample["text"]),
            }
        ],
    }
    assistant_content = {
            "role": "assistant",
            "content": [{"type": "text", "text": label_dict[dataset_name][int(sample["label"])] }],
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
        for message in messages:
            role = message["role"]
            for content in message["content"]:
                if content["type"] == "text":
                    if role == "assistant":
                        labels += content["text"] + " "

                if content["type"] == "image" and role == "user":
                    image_path = content["image"]
                    image = Image.open(image_path).convert("RGB")
                    image = image.resize((560, 560))

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = processor(image, input_text, 
                            size={"height":560, "width":560},
                            padding="max_length",
                            truncation=True,       
                            max_length=512, 
                            return_tensors="pt").to(model.device)
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
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

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct",
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
model = MllamaForConditionalGeneration.from_pretrained(
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




