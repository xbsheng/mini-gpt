from ..config import MODEL_DIR
from .gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(
    model_size="124M", models_dir=MODEL_DIR / "gpt-2"
)


print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())
# Settings: {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
# Parameter dictionary keys: dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])
