"""Train the linear head on frozen features, evaluate leave-one-generator-out, write head.json.

File score = mean of tile logits. Threshold = operating point with <= TARGET_FPR false
positives on real files, chosen from out-of-fold scores (not from training data).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

TARGET_FPR = 0.01


def file_scores(clf, scaler, feats, file_id):
    logit = clf.decision_function(scaler.transform(feats))
    ids = np.unique(file_id)
    return ids, np.array([logit[file_id == i].mean() for i in ids])


def fit(feats, label, C):
    scaler = StandardScaler().fit(feats)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(feats), label)
    return clf, scaler


def recall_at_fpr(y, s, fpr):
    neg = np.sort(s[y == 0])
    if len(neg) == 0:
        return float("nan"), float("nan")
    thr = neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))) - 1)]
    return float((s[y == 1] > thr).mean()) if (y == 1).any() else float("nan"), float(thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("../artifacts/features.npz"))
    ap.add_argument("--out", type=Path, default=Path("../artifacts/head.json"))
    ap.add_argument("--C", type=float, default=0.1)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=False)
    feats, fid, label, gen = d["feats"], d["file_id"], d["label"], d["gen"]
    # group = original file; augmented copies share it so they never straddle train/test
    files = np.unique(fid)
    f_label = np.array([label[fid == f][0] for f in files])
    f_gen = np.array([gen[fid == f][0] for f in files])
    generators = sorted(set(f_gen[f_label == 1]))

    oof_y, oof_s = [], []
    print(f"files: {len(files)} real={int((f_label==0).sum())} fake={int((f_label==1).sum())} generators={generators}")
    if len(generators) >= 2:
        real_files = files[f_label == 0]
        rng = np.random.default_rng(0)
        real_fold = dict(zip(real_files, rng.integers(0, len(generators), len(real_files))))
        for k, g in enumerate(generators):
            test_files = {f for f, fg in zip(files, f_gen) if fg == g} | {f for f in real_files if real_fold[f] == k}
            te = np.isin(fid, list(test_files))
            clf, sc = fit(feats[~te], label[~te], args.C)
            ids, s = file_scores(clf, sc, feats[te], fid[te])
            y = np.array([label[fid == i][0] for i in ids])
            auc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
            print(f"  hold out {g:20s} AUC={auc:.3f}")
            oof_y.append(y)
            oof_s.append(s)
    else:
        from sklearn.model_selection import GroupKFold

        print("fewer than 2 generators: falling back to grouped 5-fold (optimistic, not cross-generator)")
        for tr, te in GroupKFold(n_splits=5).split(feats, label, fid):
            clf, sc = fit(feats[tr], label[tr], args.C)
            ids, s = file_scores(clf, sc, feats[te], fid[te])
            oof_y.append(np.array([label[fid == i][0] for i in ids]))
            oof_s.append(s)

    y, s = np.concatenate(oof_y), np.concatenate(oof_s)
    rec, thr = recall_at_fpr(y, s, TARGET_FPR)
    print(f"out-of-fold: AUC={roc_auc_score(y, s):.3f}  recall@{TARGET_FPR:.0%}FPR={rec:.3f}  threshold(logit)={thr:.3f}")

    clf, sc = fit(feats, label, args.C)
    args.out.write_text(
        json.dumps(
            {
                "mean": sc.mean_.tolist(),
                "scale": sc.scale_.tolist(),
                "coef": clf.coef_[0].tolist(),
                "intercept": float(clf.intercept_[0]),
                "threshold": float(thr),
                "target_fpr": TARGET_FPR,
            }
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
