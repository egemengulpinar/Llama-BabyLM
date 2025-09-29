# -*- coding: utf-8 -*-
"""
Preprocessing:
- Uses only the following sources from train10M: simple_wiki, bnc_spoken, open_subtitles
- Creates corpus A (~2MB), corpus B (~10MB)
- Generates validation slices from dev/test (COMBINING THREE SOURCES)
- Trains ByteLevel-BPE tokenizer for each corpus
"""
from pathlib import Path
from typing import List, Optional
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, decoders, trainers
from tokenizers.normalizers import NFD, Lowercase, Strip
from transformers import PreTrainedTokenizerFast

# ---- Path Structure ----
ROOT = Path(__file__).resolve().parents[1]        # project root (..)
DATASET = ROOT / "dataset"
TRAIN10M = DATASET / "train10M"                   # dataset/train10M
DEV_DIR  = DATASET / "dev"                        # dataset/dev
TEST_DIR = DATASET / "test"                       # dataset/test
WORK = ROOT / "work"
OUT_DATA = WORK / "data"
MODEL_ROOT = WORK / "model"

OUT_DATA.mkdir(parents=True, exist_ok=True)
(MODEL_ROOT / "A" / "tokenizer").mkdir(parents=True, exist_ok=True)
(MODEL_ROOT / "B" / "tokenizer").mkdir(parents=True, exist_ok=True)

# ---- Parametreler ----
# Use only three sources
SOURCES = ["simple_wiki", "bnc_spoken", "open_subtitles"]
BYTES_A = 2 * 1024 * 1024     # ~2MB
BYTES_B = 10 * 1024 * 1024    # ~10MB
BYTES_EVAL = 256 * 1024       # ~256KB

def list_sources(folder: Path, include: List[str]) -> List[Path]:
    """
    Returns .train or .data files under folder containing the include keys.
    """
    paths = []
    for key in include:
        for ext in (".train", ".data"):
            p = folder / f"{key}{ext}"
            if p.exists():
                paths.append(p)
    return paths

def join_and_clip(
    sources: List[Path],
    out_file: Path,
    total_bytes: Optional[int] = None,   # None => unlimited, all files
    mode: str = "block",                 # "block" or "round_robin"
    chunk_bytes: int = 256 * 1024        # chunk size for round_robin
):
    """
    sources: files to be used sequentially (file internal ordering preserved).
    total_bytes: target total size (None means until all sources are exhausted).
    mode:
      - "block": writes allocated quota from each source in a single block (minimal mixing).
      - "round_robin": interleaves files by dividing into small chunks (light mixing).

    Note: if total_bytes is given, distributes quota equally (remaining few bytes go to first sources).
    """
    sources = [p for p in sources if p.exists()]
    assert sources, "join_and_clip: no sources found."
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print("selected join_and_clip mode :", mode)

    # if total_bytes is not specified -> write all files sequentially
    if total_bytes is None:
        if mode == "block":
            with open(out_file, "wb") as wf:
                for fp in sources:
                    with open(fp, "rb") as rf:
                        for chunk in iter(lambda: rf.read(1024 * 1024), b""):
                            wf.write(chunk)
            return
        elif mode == "round_robin":
            fhs = [open(p, "rb") for p in sources]
            try:
                alive = [True] * len(fhs)
                with open(out_file, "wb") as wf:
                    while any(alive):
                        for i, fh in enumerate(fhs):
                            if not alive[i]:
                                continue
                            chunk = fh.read(chunk_bytes)
                            if not chunk:
                                alive[i] = False
                                continue
                            wf.write(chunk)
            finally:
                for fh in fhs:
                    try: fh.close()
                    except: pass
            return

    # if total_bytes is given: distribute quota equally
    n = len(sources)
    per = total_bytes // n
    rem = total_bytes % n
    quotas = [per + (1 if i < rem else 0) for i in range(n)]

    if mode == "block":
        written = 0
        with open(out_file, "wb") as wf:
            for i, fp in enumerate(sources):
                limit = quotas[i]
                if limit <= 0:
                    continue
                with open(fp, "rb") as rf:
                    remaining = limit
                    while remaining > 0:
                        chunk = rf.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        wf.write(chunk)
                        read_len = len(chunk)
                        remaining -= read_len
                        written += read_len
                if written >= total_bytes:
                    break
        return

    elif mode == "round_robin":
        fhs = [open(p, "rb") for p in sources]
        try:
            with open(out_file, "wb") as wf:
                remaining_total = total_bytes
                done = [False] * n
                while remaining_total > 0 and not all(done):
                    progress = 0
                    for i, fh in enumerate(fhs):
                        if done[i] or quotas[i] <= 0 or remaining_total <= 0:
                            done[i] = True
                            continue
                        to_read = min(chunk_bytes, quotas[i], remaining_total)
                        if to_read <= 0:
                            done[i] = True
                            continue
                        chunk = fh.read(to_read)
                        if not chunk:
                            done[i] = True
                            continue
                        wf.write(chunk)
                        read_len = len(chunk)
                        quotas[i] -= read_len
                        remaining_total -= read_len
                        progress += read_len
                    if progress == 0:
                        break
        finally:
            for fh in fhs:
                try: fh.close()
                except: pass
        return
    else:
        raise ValueError("mode must be 'block' or 'round_robin'")

# -------------------------
# NEW: Mixed validation generation from three sources
# -------------------------
def _find_first_for_prefix(prefix: str) -> Optional[Path]:
    """Returns the first appropriate file starting with prefix under dev/ or test/."""
    for base in (DEV_DIR, TEST_DIR):
        for ext in (".dev", ".test", ".data", ".train"):
            p = base / f"{prefix}{ext}"
            if p.exists():
                return p
    return None

def make_eval_slice_mixed(out_file: Path,
                          total_bytes: int = BYTES_EVAL,
                          prefixes: List[str] = SOURCES,
                          mode: str = "block"):
    """
    Creates validation file by COMBINING THREE SOURCES.
    - total_bytes: total bytes; equal share given to each source.
    - mode: 'block' (recommended) or 'round_robin'
    """
    srcs: List[Path] = []
    for pref in prefixes:
        p = _find_first_for_prefix(pref)
        if p is not None:
            srcs.append(p)

    # Fallback: take first 3 appropriate files from dev/test directories
    if not srcs:
        pool = []
        for base in (DEV_DIR, TEST_DIR):
            pool += [p for p in sorted(base.glob("*"))
                     if p.suffix.lower() in (".dev", ".test", ".data", ".train")]
        srcs = pool[:3]

    if not srcs:
        raise FileNotFoundError("No appropriate dev/test file found for validation.")

    join_and_clip(sources=srcs, out_file=out_file, total_bytes=total_bytes, mode=mode)
    print(f"[EVAL] sources => {[p.name for p in srcs]} -> {out_file}")

def train_bpe_tokenizer(train_txt: Path, out_dir: Path, vocab_size: int):
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.Sequence([NFD(), Lowercase(), Strip()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size)
    tok.train(files=[str(train_txt)], trainer=trainer)
    tok.post_processor = processors.ByteLevel(trim_offsets=True)
    tok.decoder = decoders.ByteLevel()

    wrap = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<pad>",
    )
    wrap.save_pretrained(str(out_dir))

def main():
    # --- Corpus A: ~2MB (from three sources)
    srcs = list_sources(TRAIN10M, SOURCES)
    outA = OUT_DATA / "train_A_2MB.txt"
    if not outA.exists():
        join_and_clip(srcs, outA, BYTES_A, mode="round_robin")
        print(f"[A] train -> {outA}")

    # --- Corpus B: ~10MB (from same three sources, longer)
    outB = OUT_DATA / "train_B_10MB.txt"
    if not outB.exists():
        join_and_clip(srcs, outB, BYTES_B, mode="round_robin")
        print(f"[B] train -> {outB}")

    # --- Validation slices: FROM THREE SOURCES COMBINED
    evalA = OUT_DATA / "eval_A.txt"
    if not evalA.exists():
        make_eval_slice_mixed(out_file=evalA, total_bytes=BYTES_EVAL, mode="round_robin")
        print(f"[A] valid -> {evalA}")
    evalB = OUT_DATA / "eval_B.txt"
    if not evalB.exists():
        make_eval_slice_mixed(out_file=evalB, total_bytes=BYTES_EVAL, mode="round_robin")
        print(f"[B] valid -> {evalB}")

    # --- Tokenizer’lar
    tokA_dir = MODEL_ROOT / "A" / "tokenizer"
    tokB_dir = MODEL_ROOT / "B" / "tokenizer"
    if not (tokA_dir / "tokenizer.json").exists():
        train_bpe_tokenizer(outA, tokA_dir, vocab_size=16000)
        print(f"[A] tokenizer -> {tokA_dir}")
    if not (tokB_dir / "tokenizer.json").exists():
        train_bpe_tokenizer(outB, tokB_dir, vocab_size=32000)
        print(f"[B] tokenizer -> {tokB_dir}")

if __name__ == "__main__":
    main()
