"""
Real-LLM adversarial attack on the content classifier.

run_extended.py's text-laundering result rewrites bot tweets with a deterministic
regex (strip URLs / hashtags / ALL-CAPS) as a stand-in for an LLM adversary. This
script replaces that proxy with an actual language model: each test tweet is
rewritten by an LLM to read like a natural human post, the content features are
recomputed from the rewritten text, and the three classifiers are re-scored.

The protocol mirrors run_extended.py exactly -- same 70/30 stratified split
(seed 42), same classifiers trained on clean data, and the same severity =
fraction-of-test-tweets-rewritten sweep -- so the LLM curve in table11 is directly
comparable to the regex curve in table5.

Needs an API key: ANTHROPIC_API_KEY (default), or GEMINI_API_KEY with
--provider gemini. The whole test set is rewritten once and cached under data/,
so re-runs are free and resumable. Use --dry-run to exercise the scoring pipeline
with an identity rewrite (no API, no key).

    python run_llm_attack.py                              # Claude Haiku (default)
    python run_llm_attack.py --provider gemini --model gemini-2.5-flash
    python run_llm_attack.py --dry-run                    # pipeline smoke test, no API

Output: results/table11_llm_attack.csv
"""

import argparse
import hashlib
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from features import engineer_all, ensure_dataset1, DATA, OUT

warnings.filterwarnings("ignore")

RNG = 42
SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]

REWRITE_SYSTEM = (
    "You rewrite a single social-media post so it reads like a natural post from a "
    "real person, keeping the same general topic. Remove the tells of an automated "
    "or promotional account: excessive hashtags, links, @-mentions, ALL-CAPS, "
    "repeated punctuation, and marketing phrasing. Keep it short. Output only the "
    "rewritten post text, with no quotes or commentary."
)


def _slug(s):
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-")


def _sha(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


# --- LLM clients (provider-abstracted; only the chosen one is imported) ---
def build_rewriter(provider, model, dry_run):
    if dry_run:
        return lambda text: text  # identity: pipeline smoke test, no API

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

        def call(text):
            msg = client.messages.create(
                model=model, max_tokens=200, system=REWRITE_SYSTEM,
                messages=[{"role": "user", "content": text}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        return call

    if provider == "gemini":
        from google import genai
        client = genai.Client()  # reads GEMINI_API_KEY

        def call(text):
            resp = client.models.generate_content(
                model=model, contents=f"{REWRITE_SYSTEM}\n\nPost:\n{text}")
            return (resp.text or "").strip()
        return call

    raise SystemExit(f"unknown provider: {provider}")


def ensure_rewrites(statuses_by_id, provider, model, dry_run):
    """Rewrite every (id_str -> status) once, cached to data/. Returns id_str -> rewrite."""
    cache_path = os.path.join(DATA, f"llm_tweet_rewrites_{_slug(model)}.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                cache[r["id_str"]] = r
    rewrite = build_rewriter(provider, model, dry_run)

    todo = [(i, s) for i, s in statuses_by_id.items()
            if not (cache.get(i) and cache[i].get("text_sha") == _sha(s))]
    if todo and not dry_run:
        print(f"Rewriting {len(todo)} tweets with {provider}/{model} "
              f"({len(cache)} already cached)...")
    out = dict(cache)
    for n, (id_str, status) in enumerate(todo, 1):
        try:
            rw = rewrite(status)
        except Exception as e:  # one bad call must not abort the run
            print(f"  [{n}/{len(todo)}] id={id_str} ERROR: {type(e).__name__}; keeping original")
            rw = status
        rec = {"id_str": id_str, "text_sha": _sha(status), "rewrite": rw}
        out[id_str] = rec
        if not dry_run:
            with open(cache_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if n % 25 == 0:
                print(f"  [{n}/{len(todo)}] cached")
    return {i: r["rewrite"] for i, r in out.items()}


def load_dataset():
    df = pd.read_csv(ensure_dataset1(), encoding="utf-8", on_bad_lines="skip", low_memory=False)
    df = df.dropna(subset=["bot"]).copy()
    df["bot"] = df["bot"].astype(int)
    df = df.drop_duplicates(subset=["id_str"]).reset_index(drop=True)
    df["id_str"] = df["id_str"].astype(str)
    df["status"] = df["status"].fillna("").astype(str)
    return df


def _feat(frame, cols, content_cols, behav_cols):
    X, _, _ = engineer_all(frame)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "gemini"])
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--dry-run", action="store_true", help="identity rewrite, no API key needed")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    df = load_dataset()
    y = df["bot"].values
    _, behav_cols, content_cols = engineer_all(df)

    # Same split + clean training as run_extended.py, so table11 is comparable to table5.
    tr_df, te_df, y_tr, y_te = train_test_split(
        df, y, test_size=0.3, random_state=RNG, stratify=y)
    Xtr = _feat(tr_df, None, content_cols, behav_cols)
    models = {}
    for name, cols in [("RF-Content", content_cols),
                       ("RF-Behavioral", behav_cols),
                       ("RF-Fusion", behav_cols + content_cols)]:
        clf = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
        clf.fit(Xtr[cols].values, y_tr)
        models[name] = (clf, cols)

    # Rewrite the whole test set once (cached), then sweep severity = fraction laundered.
    rewrites = ensure_rewrites(dict(zip(te_df["id_str"], te_df["status"])),
                               args.provider, args.model, args.dry_run)

    rows = []
    for sev in SEVERITIES:
        rng = np.random.default_rng(RNG + int(sev * 100))
        mask = rng.random(len(te_df)) < sev
        laundered = te_df.copy()
        statuses = laundered["status"].tolist()
        ids = laundered["id_str"].tolist()
        for i in np.where(mask)[0]:
            statuses[i] = rewrites.get(ids[i], statuses[i])
        laundered["status"] = statuses
        Xte = _feat(laundered, None, content_cols, behav_cols)
        for name, (clf, cols) in models.items():
            proba = clf.predict_proba(Xte[cols].values)[:, 1]
            pred = (proba >= 0.5).astype(int)
            rows.append({
                "severity": sev, "model": name,
                "accuracy": accuracy_score(y_te, pred),
                "f1": f1_score(y_te, pred),
                "precision": precision_score(y_te, pred, zero_division=0),
                "recall": recall_score(y_te, pred),
                "roc_auc": roc_auc_score(y_te, proba),
            })
    table = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "table11_llm_attack.csv")
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {out_path}")

    # Headline: clean (sev 0.0) vs full LLM rewrite (sev 1.0), per model.
    print("\nROC-AUC, clean -> full LLM rewrite:")
    for name in models:
        a = table[(table.severity == 0.0) & (table.model == name)]["roc_auc"].iloc[0]
        b = table[(table.severity == 1.0) & (table.model == name)]["roc_auc"].iloc[0]
        print(f"  {name:<14} {a:.3f} -> {b:.3f}  (delta {b - a:+.3f})")


if __name__ == "__main__":
    main()
