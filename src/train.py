from dataclasses import dataclass

import tiktoken
import torch
import torch.nn.functional as F
from torch import optim

from config import DATA_DIR, DEVICE, MODEL_DIR
from dataloader import create_dataloader
from gpt import GPT_CONFIG_124M, GPTModel


@dataclass
class GPT_CONFIG_CUSTOM(GPT_CONFIG_124M):
    context_length = 256
    lr = 1e-3


GPT_2_1_MODEL_PATH = MODEL_DIR / "gpt-2.1.pth"


def train():
    config = GPT_CONFIG_CUSTOM()
    model = GPTModel(config).to(DEVICE)

    tokenizer = tiktoken.get_encoding("gpt2")
    ids = tokenizer.encode("nice to meet")
    ids = torch.tensor(ids, device=DEVICE).unsqueeze(0)

    outputs = model(ids)
    output_ids = torch.argmax(outputs, -1).squeeze().tolist()
    print(output_ids)
    text = tokenizer.decode(output_ids)
    print(text)

    with open(DATA_DIR / "the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        dataloader = create_dataloader(
            raw_text,
            batch_size=16,
            max_content=config.context_length,
            stride=int(config.context_length / 16),
            shuffle=True,
        )

        print("dataloader len:", len(dataloader))

        optimizer = optim.AdamW(model.parameters(), lr=config.lr)

        model.train()

        best_loss = float("inf")

        for epoch in range(100):
            for step, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(inputs)

                loss = F.cross_entropy(
                    outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1)
                )
                print(f"epoch {epoch} - step {step} - loss:", loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if loss < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), GPT_2_1_MODEL_PATH)
                    print(f"best_loss: {best_loss} | save success")

            outputs = model(ids)
            output_ids = torch.argmax(outputs, -1).squeeze().tolist()
            text = tokenizer.decode(output_ids)
            print(f"epoch {epoch} - text: {text}")


def pred():
    config = GPT_CONFIG_CUSTOM()
    model = GPTModel(config)
    model.load_state_dict(torch.load(GPT_2_1_MODEL_PATH))

    tokenizer = tiktoken.get_encoding("gpt2")
    start_text = "nice to meet"
    ids = tokenizer.encode(start_text)

    model.eval()
    with torch.no_grad():
        for i in range(100):
            ids_tensor = torch.tensor(ids).unsqueeze(0)
            outputs = model(ids_tensor)

            # output_ids = torch.argmax(outputs, -1).squeeze().tolist() # 贪婪采样

            prob = torch.softmax(outputs, dim=-1)
            output_id = int(torch.multinomial(prob[:, -1, :], 1).item())  # 概率采样
            ids.append(output_id)
            text = tokenizer.decode([output_id])
            if i == 0:
                text = start_text + text
            print(text, end="", flush=True)
    """
    Every effort moves you know; and I want him to enjoy himself," she said quite simply.

    I looked about the spacious white-panelled room, with its _famille-verte_ vases repeating the tones of the pale damask curtains, and its eighteenth-century pastels in delicate faded frames.

    "Has he chucked his pictures too? I haven't seen a single one in the house."

    A slight shade of constraint crossed Mrs. Gisburn's open countenance%
    """

    """
    nice to meet charming, so disarming, that one longed to cry out: "Be dissatisfied with your leisure!" as once one had longed to say: "Be dissatisfied with your work!"

    But, with the cry on my lips, my diagnosis suffered an unexpected check.

    "This is my own lair," he said, leading me into a dark plain room at the end of the florid vista. It was square and brown and leathery: no "effects"; no br%    
    """


if __name__ == "__main__":
    # train()
    pred()
