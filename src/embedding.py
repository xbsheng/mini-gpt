import tiktoken
import torch

from config import DATA_DIR
from dataloader import create_dataloader

tokenizer = tiktoken.get_encoding("gpt2")

VOCAB_SIZE = tokenizer.n_vocab
EMBEDDING_DIM = 256


embedding_layer = torch.nn.Embedding(
    num_embeddings=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
)


if __name__ == "__main__":
    with open(DATA_DIR / "the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        ids = tokenizer.encode(raw_text)

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

            embeddings = embedding_layer(inputs)
            print(embeddings.shape, embeddings)
            # embeddings.shape: torch.Size([2, 4, 256])

            break
