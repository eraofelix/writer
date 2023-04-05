#!/usr/bin/python
# -*- coding: utf-8 -*-

from pathlib import Path
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from datasets import load_dataset
import torch
from tqdm import tqdm
import json
import os
import pdb
from prompter import Prompter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split):
        txt_paths = [str(x) for x in Path(data_path).glob("**/*.txt")]
        train_content = ""
        eval_content = ""
        for txt_path in txt_paths:
            with open(txt_path, 'r') as f:
                content = f.read()
                train_content += content[:int(99*len(content)/100)] 
                eval_content += content[int(99*len(content)/100):]
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        context_length = 128
        split_content = train_content if split == "train" else eval_content
        self.outputs = tokenizer(
          split_content,  # 假如是个长度为2000的string，分割完后是500个
          truncation=True,
          padding=True,
          max_length=context_length,     # 分割，每一段最大长度是128，那么就返回4段：outputs['length']=[128, 128, 128, 116]
          return_overflowing_tokens=True, # !!!tokenize the whole input and split it into several chunks
          return_length=True,             # return the length of each created chunk automatically
          return_tensors="pt"
        )
      

    def __getitem__(self, idx):
        return self.outputs['input_ids'][idx], self.outputs['attention_mask'][idx]

    def __len__(self):
        return len(self.outputs['input_ids'])

class LoraNovelDataset():
    def __init__(self, data_path, output_path):
        if os.path.exists(output_path):
            return
        txt_paths = [str(x) for x in Path(data_path).glob("**/*.txt")]
        dicts = []
        for txt_path in txt_paths:
            with open(txt_path, 'r') as f:
                content = f.readlines()
                content = [line for line in content if line.strip()] # remove empty lines from content
                pos = 0
                while pos < len(content) - 1000:
                    concat_content = ""
                    while len(concat_content) < 256:  # accumulate short segment to at least 256 characters
                        concat_content += content[pos]
                        pos += 1
                    new_dict = {}
                    new_dict["instruction"] = "请续写下一段落"
                    new_dict["input"] = concat_content
                    new_dict["output"] = content[pos]
                    dicts.append(new_dict)

        with open(output_path, 'w') as f:
            json.dump(dicts, f, indent=2, ensure_ascii=False)


def tokenize(prompt, add_eos_token=False, max_length=512):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    if (result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < max_length
        and add_eos_token):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    if len(result["input_ids"]) < max_length:
        padding_length = max_length - len(result["input_ids"])
        result["input_ids"] = [tokenizer.pad_token_id] * padding_length + result["input_ids"]
        result["attention_mask"] = [0] * padding_length + result["attention_mask"]
        result["token_type_ids"] = [0] * padding_length + result["token_type_ids"]
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    prompter = Prompter("/Users/kun/codes/alpaca-lora/templates/alpaca.json")
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

def collate_fn(samples):
    input_ids = torch.tensor([s['input_ids'] for s in samples])
    attention_mask = torch.tensor([s['attention_mask'] for s in samples])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


if __name__ == "__main__":
    data_path = os.path.expanduser("~/codes/writer/data/")
    output_path = os.path.join(data_path, "lora_novel.json")
    dataset = LoraNovelDataset(data_path, output_path)
    data = load_dataset("json", data_files=output_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    train_val = data["train"].train_test_split(test_size=2000, shuffle=True, seed=42)
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=8, collate_fn=collate_fn)
    for i, sample in enumerate(train_dataloader):
        print(i, sample)
        break

