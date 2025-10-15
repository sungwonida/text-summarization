# occlusion_loso.py
# Usage:
#   python occlusion_loso.py --model outputs/week3/checkpoint-XXXX --texts-file docs.txt --limit 5
# Each line in docs.txt is a single source document; optionally add a tab + reference summary.

import argparse, re, os, math
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import os

def split_sentences(text):
    # Minimal, punctuation-based splitter; replace if you prefer nltk/spacy.
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]

@torch.no_grad()
def gen_summary(model, tok, text, device, max_src=512, max_new=80, num_beams=4):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_src).to(device)
    out = model.generate(**enc, max_new_tokens=max_new, num_beams=num_beams, return_dict_in_generate=True)
    return out.sequences  # (1, T)

@torch.no_grad()
def seq_logprob(model, tok, src_text, y_ids, device, max_src=512):
    # Teacher-force the SAME target y over the given source; return sum log P(y | src).
    enc = tok(src_text, return_tensors="pt", truncation=True, max_length=max_src).to(device)
    dec_inp  = y_ids[:, :-1]   # include BOS .. y_{T-1}
    target   = y_ids[:,  1:]   # y_1 .. EOS
    out = model(**enc, decoder_input_ids=dec_inp, use_cache=False, return_dict=True)
    logits = out.logits  # (1, T-1, V)
    logp = torch.log_softmax(logits, dim=-1)
    # Gather log-probs of the target tokens
    tgt_logp = logp.gather(2, target.unsqueeze(-1)).squeeze(-1)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else -1000
    mask = (target != pad_id)
    return tgt_logp.masked_select(mask).sum().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--texts-file", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--max-src", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--num-beams", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()

    # Try to import ROUGE (optional)
    rouge = None
    try:
        import evaluate
        rouge = evaluate.load("rouge")
    except Exception:
        pass

    lines = []
    with open(args.texts_file) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln: continue
            parts = re.split(r'\t|\\t', ln, maxsplit=1)
            if len(parts) == 2:
                src, ref = parts
            else:
                src, ref = ln, None
            lines.append((src, ref))
            if len(lines) >= args.limit: break

    for idx, (src_text, ref) in enumerate(lines):
        print(f"\n=== Example {idx} ===")
        sents = split_sentences(src_text)
        print(f"Sentences: {len(sents)}")

        # 1) Generate once on the full source
        y = gen_summary(model, tok, src_text, device, max_src=args.max_src,
                        max_new=args.max_new, num_beams=args.num_beams)
        base_logp = seq_logprob(model, tok, src_text, y, device, max_src=args.max_src)

        # Optional: baseline ROUGE
        base_rouge = None
        if rouge is not None and ref is not None:
            pred_txt = tok.batch_decode(y, skip_special_tokens=True)[0]
            base_rouge = rouge.compute(predictions=[pred_txt], references=[ref])

        # 2) Leave-one-sentence-out occlusion
        scores = []
        for i in range(len(sents)):
            occluded = " ".join(s for j, s in enumerate(sents) if j != i)
            occ_logp = seq_logprob(model, tok, occluded, y, device, max_src=args.max_src)
            delta_lp = base_logp - occ_logp  # higher = more important
            delta_rouge = None
            if rouge is not None and ref is not None:
                occ_y = gen_summary(model, tok, occluded, device, max_src=args.max_src,
                                    max_new=args.max_new, num_beams=args.num_beams)
                occ_txt = tok.batch_decode(occ_y, skip_special_tokens=True)[0]
                r = rouge.compute(predictions=[occ_txt], references=[ref])
                # use ROUGE-L F1 drop as a scalar summary
                delta_rouge = (base_rouge["rougeL"] if base_rouge else 0.0) - r["rougeL"]
            scores.append((i, delta_lp, delta_rouge))

        # 3) Report top sentences by influence
        scores.sort(key=lambda x: (-x[1], float("-inf") if x[2] is None else -x[2]))
        for rank, (i, dlp, dr) in enumerate(scores, 1):
            preview = sents[i][:120].replace("\n", " ")
            if dr is None:
                print(f"{rank:>2}. ΔlogP={dlp:8.3f} | sent[{i}]: {preview}")
            else:
                print(f"{rank:>2}. ΔlogP={dlp:8.3f}  ΔROUGE-L={dr:6.3f} | sent[{i}]: {preview}")

if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    main()
