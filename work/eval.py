# -*- coding: utf-8 -*-
"""
Minimal-pair evaluation (PLL) — compatible with blimp_fast
- Each .jsonl file represents a phenomenon: sentence_good / sentence_bad
- Generates overall and per-phenomenon accuracy
"""
import json, glob, random
from pathlib import Path
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
# Update blimp_fast folder path to full path (the default below matches your screenshot)
EVAL_ROOT = ROOT / "dataset" / "evaluation_data" / "blimp_fast"
MODEL_ROOT = ROOT / "work" / "model"

SEED = 42
random.seed(SEED)

def list_jsonl_files(folder: Path):
    # blimp_fast flat folder → only *.jsonl
    files = sorted(folder.glob("*.jsonl"))
    if not files:
        # backup: search in subdirectories
        files = sorted(folder.glob("**/*.jsonl"))
    return files

def load_pairs_from_file(fp: Path):
    """Reads a single phenomenon file; returns (good,bad) pairs."""
    pairs = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                g = obj.get("sentence_good") or obj.get("good") or obj.get("grammatical")
                b = obj.get("sentence_bad")  or obj.get("bad")  or obj.get("ungrammatical")
                if isinstance(g, str) and isinstance(b, str):
                    pairs.append((g.strip(), b.strip()))
            except Exception:
                continue
    return pairs

def build_eval_set(eval_root: Path, min_total=1000, max_total=None, pick_phenomena=None):
    """
    Combines samples from all phenomena.
    pick_phenomena: filter by filename parts like ['agreement', 'negation', ...] (optional).
    """
    files = list_jsonl_files(eval_root)
    if pick_phenomena:
        files = [fp for fp in files if any(key in fp.stem for key in pick_phenomena)]
    assert files, f"No jsonl files found under {eval_root}"

    data = []  
    for fp in files:
        phenom = fp.stem
        for g,b in load_pairs_from_file(fp):
            data.append({'good': g, 'bad': b, 'phenom': phenom})

    if len(data) < min_total:
        raise RuntimeError(f"Found only {len(data)} pairs < {min_total}. Add more files or lower min_total.")

    random.shuffle(data)
    if max_total:
        data = data[:max_total]
    return data

def sent_neg_logprob(model, tok, text: str, max_len=128):
    ids = tok.encode(text, add_special_tokens=False)[:max_len-1]
    if not ids:
        return 0.0
    inp = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        out = model(inp)
    logits = out.logits  # [1, T, V]
    targ = inp[:, 1:]
    logit = logits[:, :-1, :]
    loss = torch.nn.functional.cross_entropy(
        logit.reshape(-1, logit.size(-1)),
        targ.reshape(-1),
        reduction='sum'
    )
    return float(loss.detach().cpu().item())

def resolve_model_dir(variant: str) -> Path:
    """Use final_fixed if it exists, otherwise use final."""
    base = MODEL_ROOT / variant
    cand = base / "final"
    return cand if cand.exists() else None

def eval_model(model_dir: Path, pairs, max_len=128, note=""):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # overall
    correct = 0
    for p in tqdm(pairs, desc=f"PLL overall [{note}]"):
        nlp_g = sent_neg_logprob(model, tok, p["good"], max_len=max_len)
        nlp_b = sent_neg_logprob(model, tok, p["bad"],  max_len=max_len)
        correct += 1 if nlp_g < nlp_b else 0
    overall = correct / len(pairs)

    # per-phenom
    rows = []
    by_phenom = {}
    for p in pairs:
        by_phenom.setdefault(p["phenom"], []).append(p)
    for phenom, items in by_phenom.items():
        c = 0
        for p in items:
            nlp_g = sent_neg_logprob(model, tok, p["good"], max_len=max_len)
            nlp_b = sent_neg_logprob(model, tok, p["bad"],  max_len=max_len)
            c += 1 if nlp_g < nlp_b else 0
        rows.append({"phenom": phenom, "N": len(items), "acc": c/len(items)})

    df = pd.DataFrame(rows).sort_values("phenom").reset_index(drop=True)
    return overall, df

if __name__ == "__main__":

    pairs = build_eval_set(EVAL_ROOT, min_total=1000, max_total=None, pick_phenomena=None)
    print(f"Total pairs used: {len(pairs)}")

    #Model paths (takes final_fixed if it exists)
    A_DIR = resolve_model_dir("A")
    B_DIR = resolve_model_dir("B")


    accA, dfA = eval_model(A_DIR, pairs, max_len=128, note="Model-A")
    accB, dfB = eval_model(B_DIR, pairs, max_len=128, note="Model-B")


    out_overall = pd.DataFrame([{"model":"A","acc":accA},{"model":"B","acc":accB}])
    out_overall.to_csv(MODEL_ROOT / "results_summary_all.csv", index=False)

    dfA.to_csv(MODEL_ROOT / "results_by_phenom_A_all.csv", index=False)
    dfB.to_csv(MODEL_ROOT / "results_by_phenom_B_all.csv", index=False)

    print("\nOverall:")
    print(out_overall)
    print("\nPer-phenomenon files:")
    print(" -", (MODEL_ROOT / "results_by_phenom_A_all.csv").as_posix())
    print(" -", (MODEL_ROOT / "results_by_phenom_B_all.csv").as_posix())
