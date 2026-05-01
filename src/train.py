from dataclasses import dataclass

import tiktoken
import torch
import torch.nn.functional as F
from torch import optim

from config import DATA_DIR, MODEL_DIR
from dataloader import create_dataloader
from gpt import GPT_CONFIG_124M, GPTModel


@dataclass
class GPT_CONFIG_CUSTOM(GPT_CONFIG_124M):
    context_length = 256
    lr = 1e-3


def train():
    config = GPT_CONFIG_CUSTOM()
    model = GPTModel(config)

    tokenizer = tiktoken.get_encoding("gpt2")
    ids = tokenizer.encode("nice to meet")
    ids = torch.tensor(ids).unsqueeze(1)

    outputs = model(ids)
    output_ids = torch.argmax(outputs, -1).squeeze().tolist()
    print(output_ids)
    text = tokenizer.decode(output_ids)
    print(text)

    with open(DATA_DIR / "the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        dataloader = create_dataloader(
            raw_text,
            batch_size=4,
            max_content=config.context_length,
            stride=int(config.context_length / 16),
            shuffle=True,
        )

        print("dataloader len:", len(dataloader))

        optimizer = optim.AdamW(model.parameters(), lr=config.lr)

        model.train()

        for i, (inputs, targets) in enumerate(dataloader):
            outputs = model(inputs)

            loss = F.cross_entropy(
                outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1)
            )
            print(f"step {i} - loss", loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.save(model.state_dict(), MODEL_DIR / "gpt-2.1.pth")

    outputs = model(ids)
    output_ids = torch.argmax(outputs, -1).squeeze().tolist()
    print(output_ids)
    text = tokenizer.decode(output_ids)
    print(text)


if __name__ == "__main__":
    train()
