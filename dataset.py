from pathlib import Path
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
import pdb
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
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
        

