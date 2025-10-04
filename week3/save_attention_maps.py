# week3/save_attention_maps.py
import argparse, os, json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def _tidy(tokens):
    # Make token labels readable for byte-level BPE/SPM tokenizers (e.g., BART)
    out = []
    for t in tokens:
        if t.startswith("Ġ"):  # byte-level BPE space marker
            t = " " + t[1:]
        out.append(t)
    return out

@torch.no_grad()
def get_cross_attn(model, tokenizer, text, device, max_input_len, gen_kwargs):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_len).to(device)
    # 1) Generate normally (fast)
    gen = model.generate(**enc, return_dict_in_generate=True, **gen_kwargs)
    gen_ids = gen.sequences  # (1, tgt_len)

    # 2) Forward pass with teacher-forcing to collect attentions
    out = model(**enc,
                decoder_input_ids=gen_ids,
                output_attentions=True,
                return_dict=True,
                use_cache=False)

    cross = out.cross_attentions
    if not cross or all(x is None for x in cross):
        raise RuntimeError(
            "cross_attentions is None. Ensure attn_implementation='eager' "
            "and this forward pass sets output_attentions=True (it does)."
        )

    # cross_attentions: tuple(num_layers) of (batch, num_heads, tgt_len, src_len)
    return enc["input_ids"], gen_ids, out.cross_attentions

def plot_heatmap(cross, input_ids, gen_ids, tokenizer, out_png,
                 layer="last", head="mean", max_src_tokens=200):
    import matplotlib.pyplot as plt
    import numpy as np

    L = len(cross)
    layer_ix = (L - 1) if layer == "last" else int(layer)
    # shape: (num_heads, tgt_len, src_len)
    att = cross[layer_ix][0].cpu()
    if head == "mean":
        A = att.mean(0).numpy()  # (tgt_len, src_len)
        title = f"Layer {layer_ix} | mean over {att.shape[0]} heads"
    else:
        h = int(head)
        A = att[h].numpy()
        title = f"Layer {layer_ix} | head {h}"

    src_tok = tokenizer.convert_ids_to_tokens(input_ids[0][:max_src_tokens])
    tgt_tok = tokenizer.convert_ids_to_tokens(gen_ids[0])
    src_tok = _tidy(src_tok)
    tgt_tok = _tidy(tgt_tok)

    A = A[:len(tgt_tok), :len(src_tok)]

    plt.figure(figsize=(min(16, 2 + 0.25*len(src_tok)), min(12, 2 + 0.25*len(tgt_tok))))
    plt.imshow(A, aspect="auto", origin="lower")
    plt.xticks(range(len(src_tok)), src_tok, rotation=90, fontsize=7)
    plt.yticks(range(len(tgt_tok)), tgt_tok, fontsize=7)
    plt.xlabel("Source tokens")
    plt.ylabel("Generated tokens")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint dir or HF model id")
    ap.add_argument("--texts-file", required=True, help="One source document per line")
    ap.add_argument("--outdir", default="outputs/week3/attn_maps")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--max-input-len", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--layer", default="last")   # or 0..L-1
    ap.add_argument("--head", default="mean")    # or 0..H-1
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device).eval()

    # Prefer passing the backend at load time (newer HF versions)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model,
            attn_implementation="eager",   # <-- key line
        )
    except TypeError:
        # Fallback for older HF versions
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        # Try method, then config
        if hasattr(model, "set_attn_implementation"):
            try:
                model.set_attn_implementation("eager")
            except Exception:
                pass
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"

    # Optional: silence the BOS warning (safe for BART-like models)
    if getattr(model.config, "forced_bos_token_id", None) is None:
        model.config.forced_bos_token_id = 0

    model = model.to(args.device).eval()

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)

    with open(args.texts_file) as f:
        texts = [ln.strip() for ln in f if ln.strip()][:args.limit]

    for i, text in enumerate(texts):
        inp_ids, gen_ids, cross = get_cross_attn(model, tok, text, args.device, args.max_input_len, gen_kwargs)
        out_png = os.path.join(args.outdir, f"{i:03d}_layer-{args.layer}_head-{args.head}.png")
        plot_heatmap(cross, inp_ids, gen_ids, tok, out_png, layer=args.layer, head=args.head)
        print(f"[saved] {out_png}")

if __name__ == "__main__":
    main()
