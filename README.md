# Rethinking LLM-Guided Visual Question Answering: Does Deeper Knowledge Integration Really Help?

**ADCIS 2026 · Paper ID 865 · arXiv: [2610.17079](https://arxiv.org/abs/2610.17079)**

> Does deeper knowledge integration always improve visual question answering? In our data-scarce OK-VQA experiments, **late fusion achieved 39.44% exact match**, outperforming attention-level injection (31.80%) by **7.64 percentage points**.

**Authors:** Aneerban Saha, Vijay Kumar, Sameer Madhav D., Priyam Srivastava, Dhruv Bansal

[Read the paper on arXiv](https://arxiv.org/abs/2610.17079) · [Hugging Face model repository](https://huggingface.co/rishii100/knowledge-guided-vqa-models)

---

## Key Findings

| Strategy | Exact Match (%) | Edit Similarity (%) | Token F1 (%) | BERTScore |
|---|---:|---:|---:|---:|
| Baseline (no guidance) | 29.44 | 40.57 | 30.97 | 91.21 |
| **Late fusion (shallow)** | **39.44** | **48.97** | **41.34** | **92.14** |
| Attention injection (deep) | 31.80 | 42.66 | 33.33 | 91.00 |

*These are results reported in the paper on the augmented, top-500-filtered OK-VQA validation subset. They are not official full-benchmark leaderboard results.*

- **Guidance helps in this setup:** late fusion improves exact match by 10.00 percentage points over the no-guidance baseline.
- **Shallow integration wins in this data regime:** late fusion exceeds attention injection by 7.64 exact-match points and also scores higher on Edit Similarity, Token F1, and BERTScore.
- **Question type matters:** the per-question-type analysis reports that attention injection is weaker on “Who” and “Yes/No” questions, while late fusion improves over the baseline across the reported categories.
- **Training data is scarce:** the filtered dataset contains 3,144 training samples before question-paraphrase augmentation; the augmentation pipeline expands it to approximately 36,000 samples.

The findings do not mean deep integration is always inferior. They suggest that the integration strategy should be evaluated against the available data, model components, and question types.

## Overview

Knowledge-based visual question answering (VQA) requires a model to combine visual information with external facts or commonsense. This project studies how LLM-generated guidance should be integrated into a vision-language architecture under data-scarce fine-tuning.

The framework compares three variants using a shared ViT + FLAN-T5 backbone: a no-guidance baseline, decision-level late fusion, and attention-matrix injection.

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

The model concatenates the global ViT CLS representation with the guidance vector, projects the result, and passes it to FLAN-T5. The guidance affects answer generation without modifying the ViT's visual feature extraction.

**Strategy B — attention injection (deep integration)**

The guidance vector modifies the query and key projections in the last six ViT encoder layers through learned projections and gates. This directly changes attention over image patches.

### 3. Answer generation

**FLAN-T5-small** generates the answer as text. The same overall backbone is used to compare the alternative guidance-integration strategies.

## Model Components

| Component | Model | Role |
|---|---|---|
| Vision encoder | [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224) | Extract image features |
| Guidance text encoder | [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16) | Encode LLM-generated guidance |
| Answer decoder | [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small) | Generate the answer |
| LLM reasoner | Llama 3.3 70B | Generate structured guidance offline |
| Image captioner | BLIP | Produce image captions for guidance generation |

## Repository Structure

The current repository is notebook-led. The notebooks below contain the experiments and data-preparation workflows; filenames are retained as they currently exist in the repository.

```text
.
├── Copy_of_research_decoder_(1).ipynb       # Main experiments / model variants
├── Data_Creation.ipynb                      # LLM guidance and data augmentation pipeline
├── Attention_Modification.ipynb              # Deep attention-injection experiments
├── new_attention_model_final_final.ipynb     # Attention model experiments
├── okvqa_llm_augmented.json                  # Augmented OK-VQA data / guidance
├── Paper_id_865/                             # Camera-ready paper and reviewer response
├── paper graphs/                             # Figures used in the paper
├── Presentation_Script.md                    # ADCIS 2026 presentation script
├── Synopsis.pdf                              # Project synopsis
├── best_baseline_model.pth                   # Trained baseline checkpoint
├── best_fusion_model.pth                     # Trained late-fusion checkpoint
├── best_attention_model.pth                  # Trained attention-injection checkpoint
└── qavit_model.pth                           # Additional QA-ViT checkpoint
```

## Dataset and Preprocessing

- **Benchmark:** [OK-VQA](https://okvqa.allenai.org/), which contains questions requiring knowledge beyond the visible image content.
- **Images:** MS COCO Train 2014 images.
- **Answer filtering:** retain the 500 most frequent answer classes, giving 3,144 filtered training samples before augmentation (approximately six examples per class).
- **Augmentation:** paraphrase each question while keeping the image and target answer unchanged. The filtered training set expands to approximately 36,000 samples.
- **Split:** image-level 80/20 train/validation split with no image overlap, as described in the paper.

The notebooks may expect data files and image folders at specific paths. Check the path/configuration cells at the top of each notebook and adjust them to your Colab or local environment.

## Setup

The experiments were run in Google Colab on a single NVIDIA GPU (T4/L4-class hardware, depending on the run). A compatible GPU is recommended for training; inference and smaller tests may work on CPU but will be slower.

Install the core dependencies:

```bash
pip install torch torchvision transformers sentencepiece pillow python-Levenshtein tqdm scikit-learn bert-score
```

Depending on the notebook and environment, additional packages or access to model files may be needed. Check the import/setup cells before running all experiments.

## Quick Start

### 1. Open the notebook environment

Open the relevant `.ipynb` file in Google Colab or Jupyter and select a compatible runtime. The notebooks are the current source of the training and experimental workflows.

### 2. Prepare the COCO images

Download and extract the MS COCO Train 2014 images if they are not already available in your environment:

```bash
wget https://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
```

Place the extracted image directory where the notebook's dataset configuration expects it.

### 3. Prepare the augmented data

Use `okvqa_llm_augmented.json` if you intend to work with the prepared augmented guidance data. To regenerate or modify this data, inspect `Data_Creation.ipynb`; the generation workflow may require the appropriate model access and configuration.

### 4. Train or evaluate a variant

- **Baseline and late-fusion experiments:** start with `Copy_of_research_decoder_(1).ipynb` and identify the relevant baseline/fusion sections.
- **Attention injection:** use `Attention_Modification.ipynb` or `new_attention_model_final_final.ipynb` for the deep-integration implementation and experiments.

The notebook sections, file paths, checkpoints, and runtime configuration should be checked before running. The repository is currently notebook-based; it does not yet provide a single, stable command-line entry point for reproducing every experiment.

## Checkpoints

The trained checkpoints are also listed in the [Hugging Face model repository](https://huggingface.co/rishii100/knowledge-guided-vqa-models):

- `best_baseline_model.pth`
- `best_fusion_model.pth`
- `best_attention_model.pth`

To load a checkpoint, instantiate the matching model class first and then load its state dictionary. The checkpoint alone is not a standalone inference pipeline; the model definition, tokenizer/preprocessing, and compatible pretrained components must match those used during training.

## Evaluation Metrics

- **Exact Match (EM):** normalized exact string match between prediction and target.
- **Edit Similarity:** similarity derived from Levenshtein distance.
- **Token F1:** token-level precision/recall F1.
- **BERTScore:** semantic similarity measured using contextual embeddings.

For result comparisons, use the same dataset filtering, image-level split, answer normalization, and evaluation implementation across variants.

## Paper

**Title:** Rethinking LLM-Guided Visual Question Answering: Does Deeper Knowledge Integration Really Help?

- **arXiv:** [https://arxiv.org/abs/2610.17079](https://arxiv.org/abs/2610.17079)
- **Conference:** Accepted at ADCIS 2026
- **Conference paper ID:** 865
- **Primary category:** Computer Vision and Pattern Recognition (`cs.CV`)

## Citation

Please cite the arXiv version when referencing this work:

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

## License and Usage

This repository is marked as MIT licensed.
