# Opposition-Aware Graph Gating and Local Region Pooling for Object-Agnostic Zero-Shot State Recognition

This project extends the **Object-agnostic State Classification (OaSC)** framework for generalized zero-shot object-state recognition. The goal is to recognize states such as *open*, *closed*, *empty*, *filled*, *folded*, or *unplugged* without explicitly predicting the object category.

We introduce two lightweight extensions:

- **Opposition-Aware Contrastive Graph Gating (OCGG):** refines the semantic neighborhoods used by OaSC by comparing how strongly a graph concept supports a target state relative to its semantic opposite. CLIP text embeddings are used as the semantic signal before the unchanged Tr-GCN generates state classifiers.
- **Local Region Pooling (LRP):** uses the spatial `7×7` ResNet-101 feature map to identify coherent local regions containing strong state evidence. The top local responses provide a residual correction to the original global classifier score.

The original **Tr-GCN and ResNet-101 backbone remain frozen**, so the proposed components operate as lightweight refinements rather than requiring the full model to be retrained.

## Pipeline

The final system consists of two branches:

**Semantic branch**

`OaSC Knowledge Graph → CLIP-based semantic refinement → OCGG → Tr-GCN → State classifier weights`

**Visual branch**

`Input image → ResNet-101 → Global feature + Spatial feature map → Local Region Pooling`

The global and local scores are combined and then evaluated using generalized zero-shot calibration.

## Datasets

Experiments are performed on:

- **OSDD**
- **MIT-States**

The evaluation follows the generalized zero-shot setting, where both seen and unseen states appear in the test-time prediction space.

## Main Results

| Dataset | Reproduced OaSC HM | OCGG + LRP HM |
|---|---:|---:|
| OSDD | 50.07% | **51.51%** |
| MIT-States | 51.37% | **57.23%** |

The final OCGG+LRP configuration improves the reproduced OaSC baseline on both datasets, with the larger improvement observed on MIT-States.

## Final Configuration

- Visual backbone: **ResNet-101**
- Semantic classifier generator: **Tr-GCN**
- Semantic prior: **CLIP ViT-B/32 text encoder**
- LRP region size: **3×3**
- Number of selected local regions: **Top-3**
- LRP residual weight: **α = 0.20**

## Notes

OCGG specifically targets confusion between opposing states such as:

- `open ↔ closed`
- `empty ↔ filled`
- `folded ↔ unfolded`
- `plugged ↔ unplugged`

LRP is most useful when state evidence is spatially localized, while its effect can vary for states defined by more global or absence-based visual cues.

## Project Scope

This work was developed as a Pattern Recognition project studying whether **semantic graph refinement** and **localized visual evidence** can complement each other in object-agnostic zero-shot state recognition.