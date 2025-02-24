import io
import math
from collections import defaultdict
from itertools import islice

import matplotlib.pyplot as plt
import torch
from PIL import Image
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration
from transformers import TrainingArguments, Trainer

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

processor = AutoProcessor.from_pretrained(
    model_id
)

image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")]


def image_to_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def display_images(images):
    num_images = len(images)
    num_axes = math.ceil(math.sqrt(num_images))

    fig, axes = plt.subplots(num_axes, num_axes, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            image = Image.open(io.BytesIO(images[i]))
            ax.imshow(image)
            ax.set_title(f"Image {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image = example["image"]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        question = example["question"]
        answer = example["multiple_choice_answer"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch


login("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")

USE_LORA = True
USE_QLORA = False
DEVICE = "cuda"

if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj'],
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    lora_config.inference_mode = False
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config if USE_QLORA else None,
        _attn_implementation="eager",
        device_map="auto"
    )
    model.add_adapter(lora_config)
    model.enable_adapters()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print(model.get_nb_trainable_parameters())
else:
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    ).to(DEVICE)

    for param in model.model.vision_model.parameters():
        param.requires_grad = False

