import tiktoken
import torch
from tiktoken import Encoding
from torch.utils.data import DataLoader, Dataset

from config import DATA_DIR


class GPTDataset(Dataset):
    def __init__(self, tokenizer: Encoding, text: str, max_context: int, stride=1):
        self.input_ids: list[list[int]] = []
        self.target_ids: list[list[int]] = []

        ids = tokenizer.encode(text)
        for i in range(0, len(ids) - max_context, stride):
            self.input_ids.append(ids[i : i + max_context])
            self.target_ids.append(ids[i + 1 : i + max_context + 1])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index]), torch.tensor(self.target_ids[index])


def create_dataloader(
    text: str, batch_size=4, max_content=256, stride=128, shuffle=True
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(
        tokenizer=tokenizer,
        text=text,
        max_context=max_content,
        stride=stride,
    )

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


# end def


if __name__ == "__main__":
    with open(DATA_DIR / "the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        ids = tokenizer.encode(raw_text)
        print("ids:", ids[:10], "ids len:", len(ids))

        print("=" * 50)

        content_size = 4
        ids_sample = ids[50:]
        for i in range(1, content_size + 1):
            content = ids_sample[:i]
            desired = ids_sample[i]

            print(
                tokenizer.decode(content),
                "->",
                tokenizer.decode([desired]),
            )

        print("测试dataloader", "=" * 50)
        dataloader = create_dataloader(
            raw_text,
            batch_size=2,
            max_content=4,
            stride=1,
            shuffle=False,
        )

        for i, (inputs, targets) in enumerate(dataloader):
            print(inputs.shape, (inputs, targets))
            # inputs.shape: torch.Size([2, 4])
            # tensor([[  40,  367, 2885, 1464], [ 367, 2885, 1464, 1807]]),
            # tensor([[ 367, 2885, 1464, 1807], [2885, 1464, 1807, 3619]])

            break
