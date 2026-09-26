#!/usr/bin/env python3
"""Turn ERASURE's raw Adult results into a table and compare it with the paper.

ERASURE's SaveValues measure appends one JSON object per unlearner, separated by commas and
with a trailing comma, so the file is not valid JSON on its own. This wraps it in [ ] first.

For every configuration it reports F1 (macro) on test and forget, UMIA and runtime, plus the
utility and efficacy similarity terms from the LUMA metric of D'Angelo et al. (2026), computed
against the Gold model (retrain from scratch on the retain set) of the same run:

    M_U = exp(-gamma * |u_Gold - u|_1),  u = [F1_test, F1_forget]
    M_E = exp(-gamma * |e_Gold - e|_1),  e = [UMIA]
    gamma = 3

Both are 1 for Gold itself. The efficiency term (and therefore LUMA itself) is left out: in
a single-process run ERASURE's CudaPeak_MB ratchets with position in the run, so memory would
depend on config order rather than on method cost.

Usage:
    python scripts/summarize.py results/adult_1mlp_05pct_seed0.json
"""
import csv
import json
import math
import sys
from pathlib import Path

GAMMA = 3.0

# D'Angelo et al. (2026), Table 3, Adult, 5% forget: mean over 3 seeds, best setting per
# method. (F1_test, F1_forget, UMIA, time in minutes)
PAPER = {
    "Orig.": (.790, .799, .497, 2.08), "Gold": (.790, .797, .499, 2.08),
    "GD": (.790, .802, .498, 1.12), "SRL": (.789, .800, .498, 1.18),
    "NG": (.790, .798, .499, 0.06), "ANG": (.792, .800, .497, 2.21),
    "UNSIR": (.790, .803, .501, 1.24), "CF-k": (.788, .799, .500, 1.11),
    "EU-k": (.789, .799, .498, 1.98), "SalUn": (.790, .798, .498, 1.10),
    "BT": (.790, .803, .496, 2.38), "SCRUB": (.793, .800, .499, 1.10),
    "FF": (.776, .778, .499, 0.08), "SSD": (.790, .799, .498, 1.10),
}
ORDER = list(PAPER)


def load(path: Path) -> list[dict]:
    text = path.read_text().strip().rstrip(",")
    return json.loads("[" + text + "]")


def label(rec: dict) -> str:
    name, p = rec["unlearner"], rec["parameters"]
    if name == "Finetuning":
        return "CF-k" if p.get("last_trainable_layers", -1) > 0 else "GD"
    if name == "Cascade":
        first = p["sub_unlearner"][0]["class"]
        return "UNSIR" if "UNSIR" in first else "SalUn" if "Saliency" in first else "Cascade?"
    return {"Identity": "Orig.", "GoldModel": "Gold", "SuccessiveRandomLabels": "SRL",
            "NegGrad": "NG", "AdvancedNegGrad": "ANG", "eu_k": "EU-k", "BadTeaching": "BT",
            "Scrub": "SCRUB", "FisherForgetting": "FF",
            "SelectiveSynapticDampening": "SSD"}.get(name, name)


def setting(rec: dict) -> str:
    """The value the benchmark sweeps: the learning rate, or alpha for Fisher Forgetting."""
    p = rec["parameters"]
    if "alpha" in p:
        return f"alpha={p['alpha']:g}"
    if rec["unlearner"] == "Cascade":
        p = p["sub_unlearner"][-1]["parameters"]
    if rec["unlearner"] == "GoldModel":
        p = p["predictor"]["parameters"]
    lr = p.get("optimizer", {}).get("parameters", {}).get("lr")
    return f"lr={lr:g}" if lr is not None else "-"


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/adult_1mlp_05pct_seed0.json")
    records = load(path)
    gold = next(r for r in records if r["unlearner"] == "GoldModel")
    u_gold = (gold["f1_macro.test.unlearned"], gold["f1_macro.forget.unlearned"])

    rows = []
    for r in records:
        f1_t, f1_f, umia = r["f1_macro.test.unlearned"], r["f1_macro.forget.unlearned"], r["UMIA"]
        m_u = math.exp(-GAMMA * (abs(u_gold[0] - f1_t) + abs(u_gold[1] - f1_f)))
        m_e = math.exp(-GAMMA * abs(gold["UMIA"] - umia))
        rows.append({"method": label(r), "setting": setting(r), "f1_test": f1_t,
                     "f1_forget": f1_f, "umia": umia, "M_U": m_u, "M_E": m_e,
                     "minutes": r["RunTime"] / 60})
    rows.sort(key=lambda x: ORDER.index(x["method"]) if x["method"] in ORDER else len(ORDER))

    out_csv = path.with_name(path.stem + "_summary.csv")
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("| Method | Setting | F1 test | F1 forget | UMIA | M_U | M_E | Time (min) "
          "| Paper: F1 test / F1 forget / UMIA / time |")
    print("|---|---|---|---|---|---|---|---|---|")
    seen = set()
    for x in rows:
        first = x["method"] not in seen
        seen.add(x["method"])
        paper = " / ".join(f"{v:.3f}" if i < 3 else f"{v:.2f}"
                           for i, v in enumerate(PAPER[x["method"]])) if first else ""
        print(f"| {x['method'] if first else ''} | {x['setting']} | {x['f1_test']:.4f} "
              f"| {x['f1_forget']:.4f} | {x['umia']:.4f} | {x['M_U']:.3f} | {x['M_E']:.3f} "
              f"| {x['minutes']:.2f} | {paper} |")
    print(f"\n{len(records)} configurations; table written to {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
