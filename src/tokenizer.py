import re
from config import DATA_DIR


class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.word2id = vocab
        self.id2word = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> list[int]:
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        return [self.word2id[word.strip()] for word in words if word.strip()]

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

        all_words = sorted(set(preprocessed))
        vocab = {word: i for i, word in enumerate(all_words)}

        tokenizer = SimpleTokenizerV1(vocab)
        ids = tokenizer.encode(raw_text)
        print("ids:", ids[:10])
        print("words:", tokenizer.decode(ids[:10]))
