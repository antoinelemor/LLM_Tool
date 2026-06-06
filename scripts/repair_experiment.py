#!/usr/bin/env python3
"""
Official, reproducible one-vs-all training harness for a SINGLE category.

Uses the package's real trainer (get_model_class_for_name -> model.encode ->
model.run_training), so it produces the standard LLM Tool session logs under
logs/training_arena/<session_id>/ (training_metrics + training_metadata.json),
identical to a model trained through the CLI — but scriptable for autonomous,
iterative hyperparameter search.

Usage:
  python scripts/repair_experiment.py --category libertarianism \
      --base noaug --lr 2e-5 --warmup 0.0 --strategy weighted --epochs 8 \
      --tag e1-f32-weighted [--sampler] [--patience 6] [--model ...]
"""
import argparse, json, sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from llm_tool.trainers.sota_models import get_model_class_for_name
from llm_tool.utils.training_paths import get_training_logs_base
from llm_tool.utils.metadata_manager import MetadataManager

DATA = {
    "noaug": REPO / "data/youpol-themes/YouPol_SIED_training_THEMES_3to1_noaug.csv",
    "aug":   REPO / "data/youpol-themes/YouPol_SIED_training_THEMES_3to1.csv",
}


def build_binary(csv_path, category):
    """Build a 3:1-style one-vs-all binary set from the raw annotation CSV.

    Positives = rows whose annotation for `category` == 'yes'.
    Negatives = rows annotated for `category` == 'no' PLUS rows about OTHER
    themes (down-sampled to a 3:1 neg:pos ratio), matching the pipeline.
    """
    df = pd.read_csv(csv_path)

    def parse(a):
        try:
            d = json.loads(a); k = list(d.keys())[0]; return k, d[k]
        except Exception:
            return None, None
    themes, vals = zip(*df["annotation"].map(parse))
    df = df.assign(theme=themes, val=vals)

    pos = df[(df.theme == category) & (df.val == "yes")]
    neg_same = df[(df.theme == category) & (df.val == "no")]
    neg_other = df[df.theme != category]
    n_pos = len(pos)
    n_neg_target = n_pos * 3
    # prefer explicit negatives for this theme, then fill from other themes
    neg_pool = pd.concat([neg_same, neg_other])
    neg = neg_pool.sample(n=min(n_neg_target, len(neg_pool)), random_state=42)

    texts = list(pos["text"].astype(str)) + list(neg["text"].astype(str))
    labels = [1] * len(pos) + [0] * len(neg)
    out = pd.DataFrame({"text": texts, "label": labels})
    out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--base", choices=list(DATA), default="noaug")
    ap.add_argument("--model", default="almanach/camembertav2-base")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.0)
    ap.add_argument("--strategy", default="weighted")  # none|weighted|focal|asymmetric|auto
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--sampler", action="store_true")
    ap.add_argument("--freeze", action="store_true", help="linear probe: freeze encoder, train head only")
    ap.add_argument("--save", action="store_true", help="save the best model checkpoint (official deliverable)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tag", default="exp")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"repair-{args.category}-{args.tag}_{ts}"
    print(f"[harness] session_id={session_id}")
    print(f"[harness] cfg: base={args.base} model={args.model} lr={args.lr} warmup={args.warmup} "
          f"strategy={args.strategy} sampler={args.sampler} epochs={args.epochs} patience={args.patience}")

    # 1) Data
    binary = build_binary(DATA[args.base], args.category)
    n_pos = int((binary.label == 1).sum())
    print(f"[harness] dataset: {len(binary)} rows, {n_pos} pos ({100*n_pos/len(binary):.1f}%)")
    Xtr, Xval, ytr, yval = train_test_split(
        binary["text"].tolist(), binary["label"].tolist(),
        test_size=0.2, random_state=42, stratify=binary["label"].tolist(),
    )

    # 2) Model (official class selection — picks CamemBERTav2Base for camembertav2)
    model_cls = get_model_class_for_name(args.model)
    model = model_cls(model_name=args.model, max_length=512,
                      resource_recommendations={"num_workers": 0, "prefetch_factor": None, "persistent_workers": False})
    model.num_labels = 2
    model.class_names = [f"NOT_{args.category}", args.category]

    # 3) Encode (official tokenizer/dataloader path)
    train_loader = model.encode(Xtr, ytr, batch_size=args.batch_size, shuffle=True)
    val_loader = model.encode(Xval, yval, batch_size=args.batch_size, shuffle=False)

    metrics_dir = str(get_training_logs_base())

    # 4) Official metadata (so the session is reproducible like any other)
    quick_params = {
        "model_name": args.model, "epochs": args.epochs, "learning_rate": args.lr,
        "warmup_ratio": args.warmup, "reinforced_learning": False,
        "imbalance_strategy": args.strategy, "focal_gamma": args.focal_gamma,
        "imbalance_weight_source": "auto", "imbalance_weighted_sampler": bool(args.sampler),
        "early_stopping_patience": args.patience, "batch_size": args.batch_size,
    }
    try:
        mm = MetadataManager(session_id=session_id)
        mm.save_comprehensive_metadata(
            bundle=None, mode="quick",
            model_config={"selected_model": args.model, "epochs": args.epochs,
                          "training_approach": "one-vs-all", "use_reinforcement": False},
            quick_params=quick_params,
            execution_status={"status": "running", "started_at": datetime.now().isoformat()},
        )
    except Exception as e:
        print(f"[harness] metadata note: {e}")

    # 5) Train via the official core
    model.run_training(
        train_loader, val_loader,
        n_epochs=args.epochs, lr=args.lr, warmup_ratio=args.warmup,
        freeze_encoder=bool(args.freeze),
        save_model_as=("model" if args.save else None),
        reinforced_learning=False,
        metrics_output_dir=metrics_dir, session_id=session_id,
        category_name=args.category, training_approach="one-vs-all",
        multi_label=False, language="fr",
        huggingface_model_name=args.model,
        early_stopping_patience=args.patience, interactive_skip=False,
        suppress_display=True,
        imbalance_strategy=args.strategy, focal_gamma=args.focal_gamma,
        imbalance_weight_source="auto", imbalance_class_weights=None,
        imbalance_weighted_sampler=bool(args.sampler),
    )
    print(f"[harness] DONE session_id={session_id}")


if __name__ == "__main__":
    main()
