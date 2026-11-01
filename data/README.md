# Data layout

```
data/
  real/                    genuine receipts/invoices (images or PDFs), any subfolders
  fake/
    <generator-name>/      AI-generated or AI-edited files, one folder per generator
```

One folder per generator (for example `gpt-image`, `gemini`, `flux`, `sd`, `edited-inpaint`) so
`train/train_head.py` can hold a whole generator out and report whether the model generalizes
to generators it has not seen. Public sources to start from are listed in `RESEARCH.md`.

Match real and fake on format, size and compression where you can; otherwise the model can learn
the file pipeline instead of AI artifacts. Check consent and data-residency rules before using
customer documents. `data/*` is git-ignored.
