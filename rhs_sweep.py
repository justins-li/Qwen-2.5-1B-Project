"""
rhs_sweep.py — Right-hand-side sensitivity sweep.

Question: when Qwen judges "Is a + b = k correct?", does its verdict depend on k?

For a fixed set of (a, b) pairs we vary the stated result k over a window around
the true sum and record, single-turn and greedy:
  * the first-token log-odds   logodds = logit("yes") - logit("no")
  * P(yes) = sigmoid(logodds)   (softmax restricted to {yes, no})
  * the token span of k (pos_k .. pos_k_last); k is only fully known at pos_k_last
  * optionally (--generate) the greedy continuation, so "Yes, 1 + 5 = 6"-type answers are
    flagged (verdict says yes, restated result disagrees with k) as then_corrects.
    Without --generate, verdict comes from the argmax first token and then_corrects is blank.

Template (fixed; spacing matters — see tokenisation check printed at start):
    user:  "Is {a} + {b} = {k} correct? Answer yes or no."
Output CSV is resumable: rerunning skips (a, b, k) rows already present.

Usage (laptop, 1.5B):
    python rhs_sweep.py --out results/sweep_1p5b.csv
Smoke test (10 pairs, ±5):
    python rhs_sweep.py --smoke --out results/smoke.csv
7B on Colab:
    python rhs_sweep.py --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --device cuda --out results/sweep_7b.csv
"""
import argparse
import csv
import math
import os
import random
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You are a careful arithmetic checker."
USER_TEMPLATE = "Is {a} + {b} = {k} correct? Answer yes or no."
YES_VARIANTS = ["Yes", "yes", " Yes", " yes", "YES"]
NO_VARIANTS = ["No", "no", " No", " no", "NO"]
EQ_RE = re.compile(r"(-?\d+)\s*\+\s*(-?\d+)\s*=\s*(-?\d+)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default="results/sweep.csv")
    p.add_argument("--n_pairs", type=int, default=40)
    p.add_argument("--lo", type=int, default=1, help="min operand")
    p.add_argument("--hi", type=int, default=20, help="max operand")
    p.add_argument("--window", type=int, default=10, help="k in [true-window, true+window]")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--smoke", action="store_true", help="10 pairs, window 5")
    p.add_argument("--generate", action="store_true",
                   help="also run a greedy continuation (needed for then_corrects; slower). "
                        "Off by default: under this prompt the 1.5B smoke run only ever produced 'Yes.'/'No.'")
    p.add_argument("--no_generate", action="store_true", help=argparse.SUPPRESS)  # old flag, now the default
    return p.parse_args()


def single_token_ids(tok, variants):
    """Return ids of variants that are exactly one token; warn about the rest."""
    ids = []
    for v in variants:
        enc = tok.encode(v, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
        else:
            print(f"  [tokenisation] '{v}' is {len(enc)} tokens -> skipped")
    assert ids, f"none of {variants} is a single token"
    return ids


def build_inputs(tok, a, b, k):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(a=a, b=b, k=k)},
    ]
    return tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )


def locate_positions(tok, input_ids, k):
    """Positions (0-based) of the '=' token and of the first and last tokens of k in the prompt.

    For a multi-digit k the first token is only the leading digit (Qwen splits digits), so the
    model cannot know k before pos_k_last; use pos_k_last for "when is the verdict formed" analyses.
    """
    toks = tok.convert_ids_to_tokens(input_ids)
    # find the LAST '=' in the prompt (the system prompt has none, but be safe)
    eq_pos = max(i for i, t in enumerate(toks) if t.strip() in {"=", "Ġ=", "▁="} or t.endswith("="))
    k_str = str(k)
    k_pos = None
    for i in range(eq_pos + 1, len(toks)):
        if any(ch.isdigit() or ch == "-" for ch in toks[i]):
            k_pos = i
            break
    # extend from k_pos until the decoded span spells k exactly
    k_last = None
    if k_pos is not None:
        for j in range(k_pos, min(k_pos + len(k_str) + 2, len(input_ids))):
            span = tok.decode(input_ids[k_pos:j + 1]).strip()
            if span == k_str:
                k_last = j
                break
            if not k_str.startswith(span):
                break
    if k_last is None:
        print(f"  [tokenisation] WARNING: could not align tokens to k={k} "
              f"(tokens after '=': {toks[eq_pos + 1:eq_pos + 5]})")
    return eq_pos, k_pos, k_last, toks


def tokenisation_report(tok, a, b, k):
    enc = build_inputs(tok, a, b, k)
    ids = enc["input_ids"][0].tolist()
    eq_pos, k_pos, k_last, toks = locate_positions(tok, ids, k)
    print("\n=== tokenisation check ===")
    print(f"prompt tokens (user turn onward) for {a} + {b} = {k}:")
    # show only the tail (user turn + generation prompt)
    tail_start = max(0, eq_pos - 12)
    for i in range(tail_start, len(toks)):
        marks = []
        if i == eq_pos:
            marks.append("'='")
        if i == k_pos:
            marks.append("first token of k")
        if i == k_last:
            marks.append("last token of k")
        mark = ("  <-- " + ", ".join(marks)) if marks else ""
        print(f"  {i:4d}  {repr(toks[i])}{mark}")
    print(f"prompt length = {len(toks)} tokens; '=' at {eq_pos}; k spans {k_pos}..{k_last}")
    print("==========================\n")


def class_logit(logits, ids):
    """log-sum-exp over the variant ids of one class."""
    return torch.logsumexp(logits[ids], dim=0).item()


def main():
    # Windows consoles/pipes default to cp1252, which cannot print byte-level BPE tokens like 'Ċ'
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args()
    if args.smoke:
        args.n_pairs, args.window = 10, 5
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} ({args.dtype}, {args.device})")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()

    yes_ids = single_token_ids(tok, YES_VARIANTS)
    no_ids = single_token_ids(tok, NO_VARIANTS)
    print(f"yes ids {yes_ids}  no ids {no_ids}")

    # item grid: distinct (a, b) pairs, a <= b not required (order is a nuisance factor we keep random)
    all_pairs = [(a, b) for a in range(args.lo, args.hi + 1) for b in range(args.lo, args.hi + 1)]
    pairs = random.sample(all_pairs, args.n_pairs)
    a0, b0 = pairs[0]
    tokenisation_report(tok, a0, b0, a0 + b0)
    if a0 + b0 < 10:  # make sure a two-digit k is also shown
        tokenisation_report(tok, a0, b0, a0 + b0 + 10)

    fields = ["a", "b", "k", "true", "offset", "is_true", "pos_eq", "pos_k", "pos_k_last", "prompt_len",
              "logit_yes", "logit_no", "logodds", "p_yes", "first_token",
              "gen_text", "verdict", "restated_result", "then_corrects", "secs"]

    # resume
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and reader.fieldnames != fields:
                raise SystemExit(
                    f"{args.out} was written with a different column layout "
                    f"(missing: {sorted(set(fields) - set(reader.fieldnames))}). "
                    "Use a new --out file rather than appending to it.")
            for row in reader:
                done.add((int(row["a"]), int(row["b"]), int(row["k"])))
        print(f"resuming: {len(done)} rows already present")

    new_file = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new_file:
        w.writeheader()

    total = len(pairs) * (2 * args.window + 1)
    n = 0
    t_start = time.time()
    for a, b in pairs:
        true = a + b
        for k in range(true - args.window, true + args.window + 1):
            n += 1
            if k < 0:
                continue  # negative stated results are a different kind of wrongness; keep the window non-negative
            if (a, b, k) in done:
                continue
            t0 = time.time()
            enc = build_inputs(tok, a, b, k)
            enc = {kk: v.to(args.device) for kk, v in enc.items()}
            ids = enc["input_ids"][0].tolist()
            eq_pos, k_pos, k_last, _ = locate_positions(tok, ids, k)

            with torch.no_grad():
                out = model(**enc)
                last = out.logits[0, -1].float()
                ly, ln = class_logit(last, yes_ids), class_logit(last, no_ids)
                first_tok = tok.decode([int(last.argmax())])
                gen_text = ""
                if args.generate:
                    g = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                    gen_text = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

            logodds = ly - ln
            p_yes = 1.0 / (1.0 + math.exp(-logodds))
            # verdict = greedy answer: parsed from the continuation if we generated one, otherwise
            # from the argmax first token (identical to the greedy first token).
            verdict = ""
            m = re.match(r"\s*(yes|no)\b", gen_text if args.generate else first_tok, flags=re.I)
            if m:
                verdict = m.group(1).lower()
            restated, then_corrects = "", ""  # blank = not measured (no generation)
            if args.generate:
                m2 = EQ_RE.search(gen_text)
                if m2:
                    restated = int(m2.group(3))
                then_corrects = int(verdict == "yes" and restated != "" and restated != k)

            w.writerow(dict(a=a, b=b, k=k, true=true, offset=k - true, is_true=int(k == true),
                            pos_eq=eq_pos, pos_k=k_pos, pos_k_last=k_last, prompt_len=len(ids),
                            logit_yes=round(ly, 4), logit_no=round(ln, 4), logodds=round(logodds, 4),
                            p_yes=round(p_yes, 4), first_token=first_tok, gen_text=gen_text.strip(),
                            verdict=verdict, restated_result=restated, then_corrects=then_corrects,
                            secs=round(time.time() - t0, 2)))
            f.flush()
            if n % 20 == 0 or n == total:
                el = time.time() - t_start
                shown = gen_text.strip()[:40] if args.generate else first_tok
                print(f"[{n}/{total}] {a}+{b}={k}  p_yes={p_yes:.3f}  '{shown}'  "
                      f"({el/60:.1f} min elapsed)")
    f.close()
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
