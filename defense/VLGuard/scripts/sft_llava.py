import logging
import os
from contextlib import nullcontext

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
from trl.env_utils import strtobool

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets

from tqdm.rich import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from torch.utils.data import IterableDataset, Dataset, ConcatDataset

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
from data_utils import VLGuardDataset, llavadataset
from pathlib import Path

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

def apply_chat_template(messages):
    text = ''
    for i, message in enumerate(messages):
        if i == 0 and message['role'] == 'user':
            text = text + "USER: " + "<image>\n"
            for content in message['content']:
                if content['type'] == 'text':
                    text = text + content['text']
        elif i > 0 and message['role'] == 'user':
            text = text + "USER: "+ message['content'][0]['text']
        else:
            text = text + " ASSISTANT: "+ message['content'][0]['text'] + "</s>"
    
    return text

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    sft_script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.dataset_text_field = ""  # need a dummy field
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )

    ################
    # Freeze some modules
    ################

    model.requires_grad_(False)
    for p in model.multi_modal_projector.parameters():
        p.requires_grad = True

    ################
    # Create a data collator to encode text and image pairs
    ################

    def collate_fn(examples):
        input_ids = []
        labels = []
        images = []
        for example in examples:
            text = apply_chat_template(example['messages'])
            system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
            text = system_prompt + text

            input = processor(text, example['images'], return_tensors="pt")

            input_id = input['input_ids'][0]
            label = input_id.clone()

            sep = "</s>"
            rounds = text.split(sep)
            sep2 = " ASSISTANT: "
            
            cur_len = 1
            label[:cur_len]= -100
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep2)
                assert len(parts) == 2, 'The length of a conversation must be 2'
                parts[0] += sep2
                
                round_len = len(processor.tokenizer(rou, return_tensors='pt')[0]) 
                instruction_len = len(processor.tokenizer(parts[0], return_tensors='pt')[0]) - 2
            
                
                label[cur_len: cur_len+instruction_len] = - 100

                cur_len += round_len
            input_ids.append(input_id)
            labels.append(label)

            image = input['pixel_values']
            images.append(image)
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=-100)
        input_ids = input_ids[:, :processor.tokenizer.model_max_length]
        labels = labels[:, :processor.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(processor.tokenizer.pad_token_id),
        )                
        
        batch['pixel_values'] = torch.concat(images, axis=0)

        return batch


    ################
    # Dataset
    ################

    dataset1 = VLGuardDataset("./dataset/VLGuard/train_llava_format.json")
    folder_path = Path("dataset/llava-instruct-mix-vsf/data")
    file_list = ["dataset/llava-instruct-mix-vsf/data/" + f.name for f in folder_path.iterdir() if f.is_file()]
    dataset2 = load_dataset('parquet', data_files= file_list)['train']
    shuffled_dataset = dataset2.shuffle(seed=42)
    sub_dataset = shuffled_dataset.select(range(5000))

    llava_data = llavadataset(sub_dataset)
    mix_dataset = ConcatDataset([dataset1, llava_data])


    # raw_datasets = load_dataset(sft_script_args.dataset_name)
    # train_dataset = raw_datasets[sft_script_args.dataset_train_split]
    # eval_dataset = raw_datasets[sft_script_args.dataset_test_split]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            train_dataset=mix_dataset,
            # eval_dataset=eval_dataset,
            tokenizer=processor.tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
        trainer.push_to_hub()
        if Accelerator().is_main_process:
            processor.push_to_hub(training_args.hub_model_id)


