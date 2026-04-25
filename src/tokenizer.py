import re

import tiktoken

from config import DATA_DIR

UNK = "<|unk|>"
END_OF_TEXT = "<|endoftext|>"


class SimpleTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.word2id = vocab
        self.id2word = {v: k for k, v in vocab.items()}
        self.UNK_ID = vocab[UNK]

    def encode(self, text: str) -> list[int]:
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        return [self.word2id.get(word.strip(), 0) for word in words if word.strip()]

    def decode(self, ids: list[int]) -> str:
        return " ".join([self.id2word[id] for id in ids])


if __name__ == "__main__":
    with open(DATA_DIR / "the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        print("Total number of character:", len(raw_text))
        print("raw_text:", raw_text[:99])

        print("=" * 50)

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed: list[str] = [
            item.strip() for item in preprocessed if item.strip()
        ]
        print("preprocessed length", len(preprocessed))
        print("preprocessed:", preprocessed[:30])

        print("=" * 50)

        all_words = [UNK, END_OF_TEXT] + sorted(set(preprocessed))
        vocab = {word: i for i, word in enumerate(all_words)}

        tokenizer = SimpleTokenizer(vocab)
        ids = tokenizer.encode(raw_text)
        print("ids:", ids[:10])
        print("words:", tokenizer.decode(ids[:10]))

        print("测试未登录词", "=" * 50)

        # 处理 未登录词（OOV，Out-of-Vocabulary）
        text1 = "Hello, do you like tea?"
        text2 = "In the sunlit terraces of the palace."
        text = END_OF_TEXT.join((text1, text2))
        print(text)

        ids = tokenizer.encode(text)
        print("ids:", ids)
        print("words:", tokenizer.decode(ids))

        print("测试BPE", "=" * 50)
        tokenizer = tiktoken.get_encoding("gpt2")
        ids = tokenizer.encode("Akwirw ier")
        words = [tokenizer.decode([id]) for id in ids]
        print("bpe 分词结果:", words)  # ['Ak', 'w', 'ir', 'w', ' ', 'ier']
        print("bpe ids:", ids)  # [33901, 86, 343, 86, 220, 959]
