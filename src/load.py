from typing import cast

import numpy as np
import tiktoken
import torch

from config import DEVICE, MODEL_DIR
from gpt import GPT_CONFIG, GPTModel, TransformerBlock
from utils.gpt_download import download_and_load_gpt2

GPT_2_PATH = MODEL_DIR / "gpt-2"


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {{right.shape}}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt: GPTModel, params):
    gpt.pos_embedding.weight = assign(gpt.pos_embedding.weight, params["wpe"])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params["wte"])

    for b in range(len(params["blocks"])):
        block_params = params["blocks"][b]

        q_w, k_w, v_w = np.split((block_params["attn"]["c_attn"])["w"], 3, axis=-1)
        block: TransformerBlock = cast(TransformerBlock, gpt.transformer_blocks[b])

        block.attn.W_query.weight = assign(block.attn.W_query.weight, q_w.T)
        block.attn.W_key.weight = assign(block.attn.W_key.weight, k_w.T)
        block.attn.W_value.weight = assign(block.attn.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((block_params["attn"]["c_attn"])["b"], 3, axis=-1)
        block.attn.W_query.bias = assign(block.attn.W_query.bias, q_b)
        block.attn.W_key.bias = assign(block.attn.W_key.bias, k_b)
        block.attn.W_value.bias = assign(block.attn.W_value.bias, v_b)

        block.attn.out_proj.weight = assign(
            block.attn.out_proj.weight,
            block_params["attn"]["c_proj"]["w"].T,
        )
        block.attn.out_proj.bias = assign(
            block.attn.out_proj.bias,
            block_params["attn"]["c_proj"]["b"],
        )

        block.ff.layers[0].weight = assign(
            block.ff.layers[0].weight,
            block_params["mlp"]["c_fc"]["w"].T,
        )
        block.ff.layers[0].bias = assign(
            block.ff.layers[0].bias,
            block_params["mlp"]["c_fc"]["b"],
        )
        block.ff.layers[2].weight = assign(
            block.ff.layers[2].weight,
            block_params["mlp"]["c_proj"]["w"].T,
        )
        block.ff.layers[2].bias = assign(
            block.ff.layers[2].bias,
            block_params["mlp"]["c_proj"]["b"],
        )

        block.norm_1.scale = assign(block.norm_1.scale, block_params["ln_1"]["g"])
        block.norm_1.shift = assign(block.norm_1.shift, block_params["ln_1"]["b"])
        block.norm_2.scale = assign(block.norm_2.scale, block_params["ln_2"]["g"])
        block.norm_2.shift = assign(block.norm_2.shift, block_params["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

    print("✅ load weights into gpt success")


def create_gpt():
    settings, params = download_and_load_gpt2(
        model_size="124M",
        models_dir=GPT_2_PATH,
        only_load=True,
    )
    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())
    # Settings: {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
    # Parameter dictionary keys: dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])

    config = GPT_CONFIG(
        vocab_size=settings["n_vocab"],
        context_length=settings["n_ctx"],
        emb_dim=settings["n_embd"],
        n_heads=settings["n_head"],
        n_layers=settings["n_layer"],
        qkv_bias=True,
    )
    """
    OpenAI在多头注意力模块的线性层中使用了偏置向量来实现查询矩阵、键矩阵和值矩阵的计算。
    偏置向量在当前的大语言模型中不常用，因为它们并不提升建模性能，因此不是必要的。
    然而，由于我们正在使用预训练权重，因此需要匹配相应的设置以保持一致性，并启用这些偏置向量：
    """

    gpt = GPTModel(config)
    load_weights_into_gpt(gpt, params)
    gpt.to(DEVICE)

    return gpt


if __name__ == "__main__":
    gpt = create_gpt()

    tokenizer = tiktoken.get_encoding("gpt2")
    start_text = "nice to meet"
    ids = tokenizer.encode(start_text)

    gpt.eval()
    print(start_text, end="", flush=True)

    with torch.no_grad():
        for i in range(100):
            ids_tensor = torch.tensor(ids, device=DEVICE).unsqueeze(0)
            logits: torch.Tensor = gpt(ids_tensor)
            logits = logits[:, -1, :]

            # output_ids = torch.argmax(logits, -1).squeeze().tolist() # 贪婪采样

            temperature = 0.8
            logits = logits / temperature  # 温度缩放
            # 温度大于1会导致词元概率更加均匀分布，
            # 而小于1的温度将导致更加自信（更尖锐或更陡峭）的分布。

            # top-k 取值掩码
            values, indices = torch.topk(logits, 3)
            # torch.Size([1, 3]) torch.Size([1, 3])
            min_val = values[:, -1]
            logits = logits.masked_fill(
                logits < min_val,
                logits.new_full((), -torch.inf),
            )

            probs = torch.softmax(logits, dim=-1)
            output_id = int(torch.multinomial(probs, 1).item())  # 概率采样
            ids.append(output_id)
            text = tokenizer.decode([output_id])
            print(text, end="", flush=True)

            """
            nice to meet the people of the city, and I think that's the best way to do it," he said. "It's not a big deal, but I think it's important for people to be able to see that they're being treated fairly."

            He added: "I think it's really important for people to know that they're not treated like a criminal or anything like that, but they're not being treated like a criminal or anything like that. It's not a bad thing to do%   
            """
