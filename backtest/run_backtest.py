#!/usr/bin/env python3
"""Offline forecast-calibration backtest.

Runs the production Metaculus-bot LLM config (OpenRouter free model, no external
research — matches metac-bot-template/main.py as of cycle 132) against a set of
resolved binary questions and scores calibration vs outcome and vs the market
baseline.

Env: OPENROUTER_API_KEY must be set (pipe from `pass api/openrouter/key`).
Usage: OPENROUTER_API_KEY=... python3 run_backtest.py [questions.jsonl] [limit]

Outputs results.jsonl (per-question) and summary.json.
"""
import json
import math
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from statistics import mean

# prod pin is nvidia/nemotron-3-super-120b-a12b:free (cycle 132); override with
# BT_MODEL to backtest an alternative free model without touching prod.
MODEL = os.environ.get("BT_MODEL") or "nvidia/nemotron-3-super-120b-a12b:free"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ["OPENROUTER_API_KEY"]

QFILE = sys.argv[1] if len(sys.argv) > 1 else "questions.jsonl"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000

PROMPT = """You are a professional superforecaster. Today's date is {asof}.
Do NOT use any knowledge of events after {asof}.

Question: {q}

Background / resolution details:
{desc}

Think step by step (a few sentences): base rates, the current situation as of
{asof}, arguments for YES, arguments for NO, and how much time remains.

End your answer with EXACTLY this line and nothing after it:
Probability: ZZ%
where ZZ is an integer 1-99 for the probability the question resolves YES.
"""


def call_llm(prompt, tries=4):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1200,
        }
    ).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                OR_URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {KEY}",
                    "Content-Type": "application/json",
                    "X-Title": "arden-backtest",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            return d["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            wait = 15 * (attempt + 1)
            print(f"  llm retry {attempt+1}: {e} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    return None


PROB_RE = re.compile(r"Probability:\s*\**\s*(\d{1,3})\s*%", re.I)


def parse_prob(text):
    if not text:
        return None
    matches = PROB_RE.findall(text)
    if not matches:
        # last resort: any "NN%" near the end
        alt = re.findall(r"(\d{1,3})\s*%", text)
        if not alt:
            return None
        matches = alt
    p = int(matches[-1])
    return min(99, max(1, p)) / 100.0


def brier(p, y):
    return (p - y) ** 2


def logloss(p, y):
    p = min(1 - 1e-6, max(1e-6, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def main():
    qs = [json.loads(l) for l in open(QFILE)][:LIMIT]
    results = []
    fout = open("results.jsonl", "w")
    for i, q in enumerate(qs, 1):
        asof = datetime.fromtimestamp(
            q["resolution_time"] / 1000 - 3 * 86400, timezone.utc
        ).strftime("%Y-%m-%d")
        prompt = PROMPT.format(
            asof=asof, q=q["question"], desc=(q["description"] or "(none)")[:1500]
        )
        t0 = time.time()
        text = call_llm(prompt)
        p = parse_prob(text)
        if p is None:
            print(f"[{i:2d}/{len(qs)}] PARSE FAIL | {q['question'][:60]}")
            continue
        y = q["outcome"]
        row = {
            "id": q["id"],
            "question": q["question"],
            "asof": asof,
            "outcome": y,
            "bot_prob": round(p, 3),
            "market_prob": q["market_prob"],
            "bot_brier": round(brier(p, y), 4),
            "market_brier": round(brier(q["market_prob"], y), 4),
            "bot_logloss": round(logloss(p, y), 4),
            "market_logloss": round(logloss(q["market_prob"], y), 4),
            "secs": round(time.time() - t0, 1),
        }
        results.append(row)
        fout.write(json.dumps(row) + "\n")
        fout.flush()
        print(
            f"[{i:2d}/{len(qs)}] bot={p:.2f} mkt={q['market_prob']:.2f} "
            f"out={y} bB={row['bot_brier']:.3f} mB={row['market_brier']:.3f} "
            f"| {q['question'][:55]}"
        )
    fout.close()

    if not results:
        print("no results")
        return

    def cal_table(rows):
        buckets = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        out = []
        for lo, hi in buckets:
            sel = [r for r in rows if lo <= r["bot_prob"] < hi]
            if sel:
                out.append(
                    {
                        "bucket": f"{lo:.1f}-{hi:.1f}",
                        "n": len(sel),
                        "mean_pred": round(mean(r["bot_prob"] for r in sel), 3),
                        "actual_yes_rate": round(mean(r["outcome"] for r in sel), 3),
                    }
                )
        return out

    unc = [r for r in results if 0.15 <= r["market_prob"] <= 0.85]
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "n": len(results),
        "n_parse_fail": len(qs) - len(results),
        "base_rate_yes": round(mean(r["outcome"] for r in results), 3),
        "bot_brier": round(mean(r["bot_brier"] for r in results), 4),
        "market_brier": round(mean(r["market_brier"] for r in results), 4),
        "bot_logloss": round(mean(r["bot_logloss"] for r in results), 4),
        "market_logloss": round(mean(r["market_logloss"] for r in results), 4),
        "mean_bot_prob": round(mean(r["bot_prob"] for r in results), 3),
        "uncertain_subset": {
            "n": len(unc),
            "bot_brier": round(mean(r["bot_brier"] for r in unc), 4) if unc else None,
            "market_brier": round(mean(r["market_brier"] for r in unc), 4)
            if unc
            else None,
            "bot_logloss": round(mean(r["bot_logloss"] for r in unc), 4)
            if unc
            else None,
            "market_logloss": round(mean(r["market_logloss"] for r in unc), 4)
            if unc
            else None,
        },
        "calibration": cal_table(results),
    }
    json.dump(summary, open("summary.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
