# Detecting AI-generated receipts/invoices (images and PDFs)

Scope (narrowed): a binary or scored answer to "is this document image AI-generated or AI-edited?" Field matching against system data, OCR and KIE are out of scope for now and are only optional extra signals (section 6).

Figures below come from search summaries of the papers; verify against the originals before relying on them.

## 1. What the literature says

- **Generated receipts are detectable, but not by generic detectors.** GPT4o-Receipt (1,235 images: 935 GPT-4o-generated, 300 authentic, 159 merchant categories) reports Claude Sonnet 4 at 0.962 accuracy, and finds humans judge AI documents less reliably than AI photos.
- **Edits are much harder than full generation.** AIForge-Doc (AI-inpainted numeric fields on receipts and forms) reports GPT-4o zero-shot at 0.509 (chance), and a detector at 0.98 AUC in-distribution dropping sharply out of distribution (DocTamper-trained: 0.563; another: 0.751).
- **Newer generators break older detectors.** TextRich (8,602 text-rich images from GPT-Image-2, receipts included, zero-shot evaluation of five detectors) and TextFake were built because generic AIGI detectors were validated on photos, not text-heavy images.
- **Generic detectors generalize poorly across generator families.** GAN-trained detectors fail on diffusion; DIRE does not transfer to text-to-image models; UnivFD (CLIP features) degrades on harder diffusion models (about 0.82 average on unseen generators in one GenImage comparison). Detectors need retraining as generators change.
- Not covered by my searches (from general knowledge, unverified): C2PA/Content Credentials and SynthID give strong evidence when present, but metadata is easily stripped and watermarks only exist for cooperating generators, so absence proves nothing.

## 2. Recommended approach: layered, no single verdict

1. **Provenance (free, high precision, low recall)**: read C2PA manifest, XMP/EXIF, and PDF metadata (Producer, Creator, creation vs modification date). Known generator tags or editors are a strong positive. Missing metadata is only mildly suspicious (screenshots and messaging apps strip it).
2. **Trained detector on text-rich documents (core)**: fine-tune a binary classifier (real vs AI-generated vs AI-edited) on receipts/invoices.
3. **Multimodal LLM judge (optional)**: a strong VLM prompted for artifacts (garbled micro-text, inconsistent fonts, impossible totals) did well on full receipt generation but is at chance on edits, so use it only as an extra feature and for human-readable explanations.
4. **Score and route**: calibrated probability from a small stacker over steps 1-3. Mid scores go to human review.

## 3. The core detector: fine-tuning plan

- **Backbone**: a visual foundation model with a light head or LoRA. Candidates: CLIP/SigLIP/DINOv2 features + linear probe (UnivFD-style), or LoRA on a ViT. Recent work (e.g. "Simplicity Prevails", HyperDet) reports foundation-model features generalize better than small CNNs trained from scratch.
- **Resolution matters**: receipts carry fine glyph-level cues. Do not downscale to 224 px; train on high-resolution crops (total area, header, line items) and aggregate crop scores.
- **Training data**:
  - Real: your own genuine receipts and invoices, plus SROIE, CORD, WildReceipt, DocTamper source images.
  - AI-generated: GPT4o-Receipt (public on Hugging Face as Scam-AI/gpt4o-receipt), TextRich and TextFake benchmarks, plus your own generations from current models (GPT-Image, Gemini image, Flux, Stable Diffusion with receipt prompts) using local merchant styles and Indonesian formats.
  - AI-edited: AIForge-Doc style (inpaint totals/dates on genuine receipts), keep the edit mask.
- **Augmentations that mimic reality**: screenshots, JPEG recompression, WhatsApp-style resize, phone re-photographing, print and scan. Without these, detectors learn compression artifacts, not generation artifacts, and collapse in production.
- **Avoid shortcut learning**: match real and fake on size, format, compression, and background; otherwise the model detects the pipeline, not the AI.
- **Evaluation**: leave-one-generator-out (train without GPT-Image, test on it), cross-merchant splits, and post-processed test sets. Report AUC plus recall at a fixed 1% false-positive rate, since false accusations are costly.
- **Maintenance**: schedule retraining when new image generators ship; keep a rolling set of fresh generations.

## 4. Hardware and cost

LoRA or linear-probe training on a few tens of thousands of images fits one 24 GB GPU. Runpod is available if a rented GPU is needed (billable; quote the price first).

## 5. Risks and limits

- No detector is reliable against the newest generators plus post-processing; plan for human review, not auto-reject.
- Legitimate documents may be digitally created (PDF invoices exported from accounting software). Distinguish "AI-generated image" from "digitally generated PDF", which is normal.
- Adversaries can strip metadata and re-photograph. Treat results as risk signals.
- Privacy: check data-residency and consent before sending customer documents to hosted models.

## 6. Optional extras (later)

OCR plus arithmetic checks (items sum to total), duplicate-image hashing, and system-field matching are cheap and catch fraud that is not AI-related. Earlier broader notes on these were replaced in this file to keep the focus (the file was never committed, so they are not in git history).

## 7. Suggested first milestone

1. Collect 500+ genuine receipts/invoices and generate 500+ fakes with 3-4 current image generators.
2. Baseline: provenance check + zero-shot VLM judge + UnivFD-style linear probe.
3. Fine-tune the detector with realistic augmentations; run leave-one-generator-out evaluation.
4. Decide on thresholds from the false-positive budget.

## Sources

- [GPT4o-Receipt: dataset and human study](https://arxiv.org/html/2603.11442v2)
- [Scam-AI/gpt4o-receipt dataset](https://huggingface.co/datasets/Scam-AI/gpt4o-receipt)
- [TextRich: benchmark for AI-generated text-rich images from GPT-Image-2](https://arxiv.org/html/2606.19259v1)
- [TextFake: AIGI detection on text-rich images](https://arxiv.org/pdf/2606.01050)
- [AIForge-Doc](https://arxiv.org/html/2602.20569v1)
- [Can You Trust What You See? Synthetic legal evidence](https://arxiv.org/pdf/2606.07613)
- [GPT-ImgEval](https://arxiv.org/pdf/2504.02782)
- [Simplicity Prevails: foundation models for AIGI detection](https://arxiv.org/pdf/2602.01738)
- [HyperDet](https://arxiv.org/pdf/2410.06044)
- [Post-hoc distribution alignment detection](https://arxiv.org/pdf/2502.10803)
- [Bias-free training paradigm for AIGI detection](https://arxiv.org/pdf/2412.17671)
- [Few-shot learner for AIGI detection](https://arxiv.org/pdf/2501.08763)
- [DocTamper, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Qu_Towards_Robust_Tampered_Text_Detection_in_Document_Image_New_Dataset_CVPR_2023_paper.html)
- [FakeShield, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/4d4e0ab9d8ff180bf5b95c258842d16e-Paper-Conference.pdf)
