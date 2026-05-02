import json
from math import inf
from pathlib import Path
from typing import Tuple

import tiktoken
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DATA_DIR, DEVICE, MODEL_DIR
from load import create_gpt
from train import generate

INSTrUCtiON_JSON_PATH = DATA_DIR / "instruction/instruction-data.json"

END_OF_TEXT_ID = 50256

LOSS_IGNORE_INDEX = -100

IS_QUESTION_MASK = True


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, json_path: Path):
        tokenizer = tiktoken.get_encoding("gpt2")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            self.encode_ids = []

            for entry in data:
                input_text = format_input(entry)
                response_text = f"\n\n### Response:\n{entry['output']}"

                self.encode_ids.append(
                    (
                        tokenizer.encode(input_text),
                        tokenizer.encode(response_text),
                    )
                )

    def __len__(self):
        return len(self.encode_ids)

    def __getitem__(self, i: int):
        return self.encode_ids[i]


def collate_fn(batch: list[Tuple[list[int], list[int]]]):
    max_len = max(len(input_ids + response_ids) for input_ids, response_ids in batch)

    inputs = []
    targets = []

    for input_ids, response_ids in batch:
        full_ids = input_ids + response_ids
        full_len = len(full_ids)

        input = full_ids + [END_OF_TEXT_ID] * (max_len - full_len)

        # target = input[1:] + [END_OF_TEXT_ID]
        target = (
            input[1:full_len]
            + [END_OF_TEXT_ID]
            + [LOSS_IGNORE_INDEX] * (max_len - full_len)
        )

        # IS_QUESTION_MASK 控制 问题部分是否参与loss计算
        mask_len = len(input_ids) - 1 if IS_QUESTION_MASK else 0
        target = [LOSS_IGNORE_INDEX] * mask_len + target[mask_len:]

        inputs.append(input)
        targets.append(target)

    return torch.tensor(inputs, device=DEVICE), torch.tensor(targets, device=DEVICE)


if __name__ == "__main__":
    ds = InstructionDataset(INSTrUCtiON_JSON_PATH)

    dataloader = DataLoader(
        ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # gpt = create_gpt(model_size="355M")
    gpt = create_gpt(model_size="124M", only_load=True)

    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-5)

    best_loss = inf

    for echo in range(50):
        for i, (inputs, targets) in enumerate(dataloader, 1):
            gpt.train()
            optimizer.zero_grad()

            logits: torch.Tensor = gpt(inputs)
            loss = cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=LOSS_IGNORE_INDEX,
            )
            loss_val = loss.item()
            print(f"echo {echo + 1} step {i} - {loss_val}")

            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(gpt.state_dict(), MODEL_DIR / "gpt-2-instruction")

            loss.backward()
            optimizer.step()

        print(f"echo {echo + 1}", "=" * 50)
        generate(
            gpt,
            (
                "Below is an instruction that describes a task."
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                "Evaluate the following phrase by transforming it into the spelling given.\n\n"
                "### Input:"
                "freind --> friend\n"
            ),
        )
