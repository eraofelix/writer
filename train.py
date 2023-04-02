from pathlib import Path
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
import pdb
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from dataset import MyDataset

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
data_path = "/Users/kun/codes/data/fantasy_novel"

context_length = 128
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

train_dataset = MyDataset(data_path, split="train")
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=12)

eval_dataset = MyDataset(data_path, split="eval")
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
for epoch in tqdm(range(100)):
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

    model.eval()
    eval_loss = 0
    for batch in tqdm(eval_dataloader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = input_ids.clone().detach()
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            eval_loss += outputs[0].mean().item()
    eval_loss /= len(eval_dataloader)
    print(f" Epoch {epoch} eval_loss: {eval_loss}")
    model.train()
    torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")
