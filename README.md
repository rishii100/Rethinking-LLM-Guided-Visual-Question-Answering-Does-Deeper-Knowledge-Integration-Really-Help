# Rethinking LLM-Guided Visual Question Answering: Does Deeper Knowledge Integration Really Help?

arXiv: [2610.17079](https://arxiv.org/abs/2610.17079)**

Does deeper integration of LLM-generated knowledge always improve Visual Question Answering (VQA)? This project compares decision-level late fusion with attention-matrix injection in a shared ViT + FLAN-T5 framework under data-scarce fine-tuning.

## Key Findings

| Strategy | Exact Match (%) | Edit Similarity (%) | Token F1 (%) | BERTScore |
|---|---:|---:|---:|---:|
| Baseline (no guidance) | 29.44 | 40.57 | 30.97 | 91.21 |
| **Late fusion (shallow)** | **39.44** | **48.97** | **41.34** | **92.14** |
| Attention injection (deep) | 31.80 | 42.66 | 33.33 | 91.00 |

*These are the paper's reported results on an augmented, top-500-filtered OK-VQA validation subset. They are not official full-benchmark leaderboard results.*

- **Late fusion improves exact match by 10.00 percentage points** over the no-guidance baseline.
- **Late fusion beats attention injection by 7.64 exact-match points** and scores higher on the other reported metrics in this experimental setting.
- **Question type matters:** the paper's per-question-type analysis finds attention injection particularly weak on “Who” and “Yes/No” questions, while late fusion improves over the baseline across the reported categories.
- **Data augmentation:** the filtered set contains 3,144 samples before question-paraphrase augmentation. The Hugging Face dataset card reports 36,442 augmented samples.

These findings do not mean deep integration is always inferior. They suggest that integration strategy should be evaluated in the context of the data regime, model components, and question types.

## Research Paper and Resources

- 📄 **Paper:** [arXiv:2610.17079](https://arxiv.org/abs/2610.17079)
- 🤗 **Augmented dataset:** [OK-VQA LLM-Augmented Dataset on Hugging Face](https://huggingface.co/datasets/rishii100/okvqa-llm-augmented)
- 🤖 **Trained checkpoints:** [Knowledge-Guided VQA Models on Hugging Face](https://huggingface.co/rishii100/knowledge-guided-vqa-models)
- 💻 **Code and experiments:** [This GitHub repo](https://github.com/rishii100/Rethinking-LLM-Guided-Visual-Question-Answering-Does-Deeper-Knowledge-Integration-Really-Help)

## Overview

Knowledge-based VQA requires a model to combine image content with external facts and commonsense. This project studies how LLM-generated guidance should be integrated into a vision-language architecture under data-scarce fine-tuning.

The framework compares three variants built around a shared ViT + FLAN-T5 backbone: a no-guidance baseline, decision-level late fusion, and attention-matrix injection.

## Architecture

### 1. Offline captioning and guidance generation

1. **BLIP** generates a caption for each image.
2. **Llama 3.3 70B** receives the question and caption and generates structured guidance:
   - `visual_focus`: concepts relevant to the question;
   - `distractors`: image elements that should be ignored;
   - `knowledge`: external facts needed to answer.
3. A frozen **CLIP text encoder** embeds the guidance, which is projected into a guidance vector, **P**.

Guidance is generated offline and cached, so the trained VQA model does not need to call the LLM for every inference request.

### 2. Integration strategies

**Baseline — no guidance**  
The ViT encodes the image. Its global CLS representation is projected and passed as a prefix alongside the question representation to the FLAN-T5 answer generator.

**Strategy A — late fusion (shallow integration)**  
The global ViT CLS representation is combined with the guidance vector and projected before answer decoding. The guidance influences answer generation without changing the ViT's visual feature extraction.

**Strategy B — attention injection (deep integration)**  
The guidance vector modifies the query and key projections in the last six ViT encoder layers through learned projections and gates, changing attention over image patches.

### 3. Answer generation

**FLAN-T5-small** generates the answer as text. The shared backbone helps make the integration strategy the main variable in the comparison.

## Model Components

| Component | Model | Role |
|---|---|---|
| Vision encoder | [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224) | Extract image features |
| Guidance text encoder | [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16) | Encode LLM-generated guidance |
| Answer decoder | [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small) | Generate the answer |
| LLM reasoner | Llama 3.3 70B | Generate structured guidance offline |
| Image captioner | BLIP | Produce image captions for guidance generation |

## Repository Structure

The repository is currently notebook-led. The following files are present in the repository root:

```text
.
├── Data_Creation.ipynb          # LLM guidance generation and data augmentation
├── Attention_Modification.ipynb # Deep attention-injection experiments
├── Final Code.ipynb             # Main model experiments
├── okvqa_llm_augmented.json     # Augmented OK-VQA data / guidance
├── LICENSE
└── README.md
```

The trained checkpoints are linked from the Hugging Face model repository rather than listed as files in this structure.

## Dataset and Preprocessing

- **Benchmark:** [OK-VQA](https://okvqa.allenai.org/), which contains questions that require knowledge beyond the visible image content.
- **Images:** MS COCO 2014 Train images.
- **Answer filtering:** the top 500 answer classes produce 3,144 filtered training samples before augmentation (approximately six samples per class).
- **Augmentation:** questions are paraphrased while the associated image and target answer are intended to remain unchanged. The dataset card reports 36,442 augmented samples.
- **Split:** the paper describes an image-level 80/20 train/validation split with no image overlap.

### Load the hosted dataset

Install the Hugging Face `datasets` library:

```bash
pip install datasets
```

Then load the dataset:

```python
from datasets import load_dataset

dataset = load_dataset("rishii100/okvqa-llm-augmented")
sample = dataset["train"][0]

print("Question:", sample["question"])
print("Answer:", sample["answer"])
print("Visual focus:", sample["visual_focus"])
print("Knowledge:", sample["knowledge"])
```

Refer to the [Hugging Face dataset card](https://huggingface.co/datasets/rishii100/okvqa-llm-augmented) for the schema and dataset-specific usage information.

The notebooks may expect data files and image folders at specific paths. Check the path/configuration cells and adjust them for your Colab or local environment.

## Setup

The experiments were run in Google Colab on a single NVIDIA GPU (T4/L4-class hardware, depending on the run). A compatible GPU is recommended for training; inference and smaller tests may run on CPU but will be slower.

Install the core dependencies:

```bash
pip install torch torchvision transformers sentencepiece pillow python-Levenshtein tqdm scikit-learn bert-score datasets
```

Additional packages or model access may be required depending on the notebook and environment. Review the import/setup cells before running the experiments.

## Quick Start

### 1. Open the notebooks

Open the relevant `.ipynb` file in Google Colab or Jupyter and select a compatible runtime. The notebooks are the current source of the training and experimental workflows.

### 2. Prepare the COCO images

Download and extract the MS COCO Train 2014 images if they are not already available:

```bash
wget https://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
```

Place the extracted image directory where the notebook's dataset configuration expects it.

### 3. Prepare the augmented data

Use `okvqa_llm_augmented.json` or load the hosted [Hugging Face dataset](https://huggingface.co/datasets/rishii100/okvqa-llm-augmented). To regenerate or modify the augmentation, inspect `Data_Creation.ipynb`; the workflow may require the appropriate model access and configuration.

### 4. Train or evaluate a variant

- **Main model experiments:** start with `Final Code.ipynb` and inspect its sections for the intended baseline and fusion runs.
- **Attention injection:** use `Attention_Modification.ipynb` for the deep-integration experiments.
- **Data and guidance preparation:** use `Data_Creation.ipynb`.

Notebook sections, file paths, checkpoints, and runtime configuration should be checked before running. The repository does not currently provide a single, stable command-line entry point for reproducing every experiment.

## Checkpoints

Trained checkpoints are available from the [Hugging Face model repository](https://huggingface.co/rishii100/knowledge-guided-vqa-models):

- `best_baseline_model.pth` — baseline (no guidance)
- `best_fusion_model.pth` — late fusion
- `best_attention_model.pth` — attention injection

A checkpoint is not a standalone inference pipeline. Instantiate the corresponding model class and ensure that the model definition, tokenizer/preprocessing, and pretrained components match those used during training.

## Evaluation Metrics

- **Exact Match (EM):** normalized exact string match between prediction and target.
- **Edit Similarity:** similarity derived from Levenshtein distance.
- **Token F1:** token-level precision/recall F1.
- **BERTScore:** semantic similarity measured using contextual embeddings.

For comparisons, keep dataset filtering, the image-level split, answer normalization, and evaluation implementation consistent across variants.

## Citation

If you use this work, please cite the arXiv version:

```bibtex
@misc{saha2026rethinkingvqa,
  title         = {Rethinking LLM-Guided Visual Question Answering: Does Deeper Knowledge Integration Really Help?},
  author        = {Saha, Aneerban and Kumar, Vijay and Madhav D., Sameer and Srivastava, Priyam and Bansal, Dhruv},
  year          = {2026},
  eprint        = {2610.17079},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.17079}
}
```

**Publication status:** Accepted at ADCIS 2026.

## License and Third-Party Resources

This repository includes a `LICENSE` file declaring the MIT License. That notice does not override the terms or rights associated with third-party models, datasets, annotations, or images. Check the relevant upstream terms before redistributing or using third-party assets, including the OK-VQA-derived data and MS COCO images.
