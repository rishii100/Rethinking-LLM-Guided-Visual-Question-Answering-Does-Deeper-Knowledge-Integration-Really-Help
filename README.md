# Rethinking LLM-Guided Visual Question Answering: Does Deeper Knowledge Integration Really Help?

**Paper ID 865 · ADCIS 2026**

> We show that shallow late fusion of LLM-generated guidance outperforms deep attention-level injection by **+7.6 points** exact match on OK-VQA — challenging the assumption that deeper integration is always better.

**Authors:** Vijay Kumar, Aneerban Saha, Sameer Madhav, Priyam Srivastava, Dhruv Bansal

---

## 📌 Key Findings

| Model | Exact Match | Edit Similarity | BERTScore |
|---|---|---|---|
| Baseline (no guidance) | 29.4% | 49.6% | — |
| **Late Fusion (shallow)** ✅ | **39.4%** | **56.4%** | **highest** |
| Attention Modification (deep) | 31.8% | — | below baseline |

- **LLM guidance helps** — +10 points of exact match over no-guidance baseline
- **Shallow beats deep** on every single metric
- Deep injection disturbs pretrained ViT representations, especially hurting "who" and "yes/no" questions

---

## 🏗️ Architecture

The system has three stages:

### 1. Offline LLM Guidance Generation
- **BLIP** generates image captions
- **Llama 3.3 70B** reasons over each (question, caption) pair and produces structured JSON:
  - `visual_focus` — objects relevant to the question
  - `distractors` — objects to ignore
  - `knowledge` — external facts needed to answer
- A frozen **CLIP** text encoder converts these into a single guidance vector **P**

### 2. Two Integration Strategies Compared

**Strategy A — Late Fusion (Shallow):**
The frozen ViT extracts image features. The CLS token is concatenated with guidance vector P, projected, and fed as a prefix token to FLAN-T5 which generates the answer. The ViT never sees the guidance.

**Strategy B — Attention Modification (Deep):**
The guidance vector P is projected and injected into the self-attention Q/K matrices of the last 6 ViT encoder layers via learnable gates, modifying how the model attends to image patches.

### 3. Answer Decoder
- **FLAN-T5** (seq2seq) generates the answer as text

---

## 🗂️ Repository Structure

```
├── Copy_of_research_decoder_(1).ipynb  # Main experiment notebook (all variants)
├── Data_Creation.ipynb                 # LLM augmentation pipeline
├── Attention_Modification.ipynb        # Deep attention injection experiments
├── new_attention_model_final_final.ipynb # Final attention model
├── okvqa_llm_augmented.json            # LLM-augmented OK-VQA dataset
├── Paper_id_865/                       # Camera-ready paper + reviewer response
├── paper graphs/                       # Figures used in the paper
├── Presentation_Script.md              # ADCIS 2026 presentation script
├── Synopsis.pdf                        # Project synopsis
├── best_baseline_model.pth             # Trained baseline weights
├── best_fusion_model.pth               # Trained late fusion weights
├── best_attention_model.pth            # Trained attention modification weights
└── qavit_model.pth                     # QA-ViT model weights
```

---

## 📊 Dataset

- **OK-VQA** — 14,000+ questions requiring external knowledge
- Filtered to top 500 answer classes → **3,144 samples** (≈6 per class)
- **LLM paraphrase augmentation** expands to **~36,000 samples**
- **Image-disjoint split** — no image appears in both train and val
- Images from **COCO Train 2014**

---

## 🧩 Model Components

| Component | Model | Role |
|---|---|---|
| Vision Encoder | `google/vit-base-patch16-224` | Image feature extraction (frozen) |
| Text Guidance Encoder | `openai/clip-vit-base-patch16` | Encode visual_focus + knowledge into guidance vector |
| Answer Decoder | `google/flan-t5-small` | Generate answer text |
| LLM Reasoner | Llama 3.3 70B | Offline guidance generation |
| Captioner | BLIP | Offline image captioning |

---

## ⚙️ Setup & Requirements

```bash
pip install torch torchvision transformers pillow python-Levenshtein tqdm scikit-learn
```

**Hardware:** All experiments run on a single Google Colab GPU (T4/L4).

---

## 🚀 Quick Start

1. **Download COCO images:**
   ```bash
   wget http://images.cocodataset.org/zips/train2014.zip
   unzip train2014.zip
   ```

2. **Prepare augmented data:**
   The `okvqa_llm_augmented.json` contains pre-generated LLM guidance. To regenerate, run `Data_Creation.ipynb`.

3. **Train & evaluate:**
   Open `Final Code.ipynb` in Google Colab and run the desired section:
   - **"# Baseline"** — ViT + T5, no guidance
   - **"# Using FLAN-T5-LARGE Decoder"** — Late fusion variant
   - **"## Attention modification"** — Deep attention injection
   - **"# Residual Connections"** — Residual guidance injection

---

## 📈 Evaluation Metrics

- **Exact Match** — strict string equality after normalization
- **Edit Similarity** — 1 − (Levenshtein distance / max length)
- **Token F1** — token-level precision/recall
- **BERTScore** — semantic similarity via BERT embeddings

---

## 📄 Citation

```bibtex
@inproceedings{kumar2026rethinking,
  title={Rethinking LLM-Guided Visual Question Answering: Does Deeper Knowledge Integration Really Help?},
  author={Kumar, Vijay and Saha, Aneerban and Madhav, Sameer and Srivastava, Priyam and Bansal, Dhruv},
  booktitle={Proceedings of ADCIS 2026},
  year={2026}
}
```

---

## 📝 License

This repository is released for academic research purposes.
