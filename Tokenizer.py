from pathlib import Path
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
import pdb
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

data_path = "/Users/kun/codes/data/fantasy_novel"
paths = [str(x) for x in Path(data_path).glob("**/*.txt")]

content = None
with open(paths[0], 'r') as f:
    content = f.read()
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
context_length = 128
outputs = tokenizer(
  content,  # 假如是个长度为2000的string，分割完后是500个
  truncation=True,
  padding=True,
  max_length=context_length,     # 分割，每一段最大长度是128，那么就返回4段：outputs['length']=[128, 128, 128, 116]
  return_overflowing_tokens=True, # !!!tokenize the whole input and split it into several chunks
  return_length=True,             # return the length of each created chunk automatically
  return_tensors="pt"
)
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, outputs):
        self.outputs = outputs

    def __getitem__(self, idx):
        return self.outputs['input_ids'][idx], self.outputs['attention_mask'][idx]

    def __len__(self):
        return len(self.outputs['input_ids'])
        
dataset = MyDataset(outputs)



train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
for epoch in tqdm(range(10)):
    for batch in tqdm(train_dataloader, total=len(train_dataloader)):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = input_ids.clone().detach()
        labels[labels == tokenizer.pad_token_id] = -100
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f" Epoch {epoch} loss: {loss.item()}") 
