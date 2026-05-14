# Learning EEG Representations for Sleep Deprivation Classification and Sleepiness Prediction

This repository contains our course project on using resting-state EEG for two related tasks:

1. **Sleep deprivation classification**: classify EEG epochs as **normal sleep (NS)** vs. **sleep deprivation (SD)**
2. **Sleepiness prediction**: predict session-level **Stanford Sleepiness Scale (SSS)** and **Karolinska Sleepiness Scale (KSS)** scores

The core goal is to study whether EEG representations learned for sleep deprivation classification are also useful for downstream subjective sleepiness prediction.

Link to Drive Folder containing raw data, preprocessed data, group presentation, group video: [https://drive.google.com/drive/folders/1meAulHb0yytaVB1TZRkgO1hG_Lgp4cI_?usp=sharing]

## What this repo includes

- EEG preprocessing pipeline
- Phase 1 models for NS vs. SD classification
- Phase 2 pipelines for sleepiness prediction:
  - **Phase 1 feature-based models** using pretrained representations
  - **Direct end-to-end ordinal prediction** from EEG
  - **Multi-task learning** for joint classification and sleepiness prediction

## Main approach

### Phase 1: NS vs. SD classification

We train deep models on preprocessed EEG epochs and compare:

- Residual EEG CNN
- 2-Branch EEG CNN
- DeiT-Tiny
- ViT-Small

The **Residual EEG CNN** is then used as the main encoder for downstream experiments.

### Phase 2: Sleepiness prediction

We evaluate three settings:

- **Phase 1 feature-based prediction**  
  Extract epoch-level features from the trained Residual EEG CNN and train session-level models such as Ridge, MLP, GRU, and LSTM.

- **Direct ordinal prediction**  
  Train EEG models directly on SSS or KSS using ordinal supervision.

- **Multi-task learning**  
  Jointly learn NS vs. SD classification and sleepiness prediction using a shared encoder.

## Notes

- The project uses subject-wise splitting to avoid leakage across train, validation, and test sets.
- Pretrained classification checkpoints used for downstream feature extraction are stored in models/.
- Experiments are provided as notebooks under code/.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Team Members

- Rushendra Sidibomma
- Chinmay Arvind
- Samarth Kumar Samal
- Arno Benzigar
