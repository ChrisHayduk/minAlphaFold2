# minAlphaFold2

A minimal, pedagogical PyTorch reimplementation of [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2).

Inspired by Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT).

<p align="center">
  <img src="assets/af2_img.png" alt="AlphaFold 2 architecture diagram" width="600">
  <br>
  <em>Diagram of AlphaFold 2 as published in DeepMind's blogpost in November 2020.</em>
</p>

## Philosophy

- **Pure PyTorch.** Every layer is built from `nn.Linear`, `nn.LayerNorm`, `torch.einsum`, and standard activations. No external ML libraries, no hidden abstractions.
- **1-to-1 mapping to the supplement.** Each module corresponds directly to numbered algorithms in the [AlphaFold2 supplementary information](https://www.nature.com/articles/s41586-021-03819-2#Sec20). Comments reference specific algorithm and line numbers throughout.
- **Designed to be read and modified.** The entire model fits in ~3,500 lines across 9 modules. If you can read PyTorch, you can read this.

## File Structure

```
minalphafold/
    a3m.py                # Minimal A3M parsing and tokenization
    mmcif.py              # Minimal mmCIF atom-site parsing to atom14 coordinates
    geometry.py           # Rigid frames, torsions, pseudo-beta helpers for supervision
    data.py               # Processed OpenProteinSet dataset, crops, collation, feature builders
    model.py              # Top-level AlphaFold2 module, recycling loop, ensemble averaging
    embedders.py          # Input embedding, relative position encoding, all attention/update modules
    evoformer.py          # Evoformer block, MSA row attention with pair bias
    structure_module.py   # Structure module, IPA, backbone update, all-atom coordinate generation
    heads.py              # Distogram, pLDDT, masked MSA, TM-score, experimentally resolved heads
    losses.py             # FAPE (backbone + all-atom), torsion angle, pLDDT, distogram, MSA, structural violation losses
    utils.py              # Dropout (row/column-wise), distance binning, recycling distogram
    residue_constants.py  # Amino acid chemical data (frames, bond lengths, VDW radii, torsion masks)
    trainer.py            # Training loop (placeholder)
scripts/
    download_openproteinset.py   # Minimal OpenProteinSet downloader/setup helper
    preprocess_openproteinset.py # Raw OpenProteinSet -> per-chain NPZ caches
tests/
    test_shapes.py        # Core shape and semantic tests
    test_a3m.py           # A3M parser tests
    test_mmcif.py         # mmCIF parser tests
    test_geometry.py      # Geometry helper tests
    test_data_pipeline.py # Dataset, preprocessing, and end-to-end batch tests
```

## Supplement Algorithm Mapping

| Algorithm | Description | Location |
|-----------|-------------|----------|
| 1 | MSA Block Deletion | `data.py: block_delete_msa` |
| 2 | Inference | `model.py: AlphaFold2.forward` |
| 3 | Input Embedder | `embedders.py: InputEmbedder` |
| 4 | Relative Position Encoding | `embedders.py: RelPos` |
| 5 | One-hot Nearest Bin | `utils.py: one_hot_nearest` |
| 6 | Evoformer Stack | `evoformer.py: Evoformer` |
| 7 | MSA Row Attention with Pair Bias | `evoformer.py: MSARowAttentionWithPairBias` |
| 8 | MSA Column Attention | `embedders.py: MSAColumnAttention` |
| 9 | MSA Transition | `embedders.py: MSATransition` |
| 10 | Outer Product Mean | `embedders.py: OuterProductMean` |
| 11 | Triangle Multiplication (Outgoing) | `embedders.py: TriangleMultiplicationOutgoing` |
| 12 | Triangle Multiplication (Incoming) | `embedders.py: TriangleMultiplicationIncoming` |
| 13 | Triangle Attention (Starting Node) | `embedders.py: TriangleAttentionStartingNode` |
| 14 | Triangle Attention (Ending Node) | `embedders.py: TriangleAttentionEndingNode` |
| 15 | Pair Transition | `embedders.py: PairTransition` |
| 16 | Template Pair Stack | `embedders.py: TemplatePair` |
| 17 | Template Pointwise Attention | `embedders.py: TemplatePointwiseAttention` |
| 18 | Extra MSA Stack | `embedders.py: ExtraMsaStack` |
| 19 | MSA Column Global Attention | `embedders.py: MSAColumnGlobalAttention` |
| 20 | Structure Module | `structure_module.py: StructureModule` |
| 21 | Rigid Frames from Three Points | `geometry.py: backbone_frames` |
| 22 | Invariant Point Attention (IPA) | `structure_module.py: InvariantPointAttention` |
| 23 | Backbone Update | `structure_module.py: BackboneUpdate` |
| 24 | Compute All Atom Coordinates | `structure_module.py: compute_all_atom_coordinates` |
| 25 | Rigid-group Frames from Torsions | `structure_module.py: make_rot_x`, `compose_transforms` |
| 26 | Rename Symmetric Ground Truth Atoms | `losses.py: AlphaFoldLoss.forward` (lines 83–94, computes `true_torsion_angles_alt`) |
| 27 | Torsion Angle Loss | `losses.py: TorsionAngleLoss` |
| 28 | FAPE (Backbone) | `losses.py: BackboneFAPE` |
| 28 | FAPE (All-Atom) | `losses.py: AllAtomFAPE` |
| 29 | PLDDT Head | `heads.py: PLDDTHead` & `losses.py: PLDDTLoss` |
| 30 | Inference with Recycling | `model.py: AlphaFold2.forward` (fixed number of cycles during inference) |
| 31 | Training with Recycling | `model.py: AlphaFold2.forward` (random cycle sampling) |
| 32 | Recycling Embedder | `model.py: AlphaFold2.forward` (recycle norms + distance bins) |

## Key Design Decisions

- **Pure PyTorch primitives.** `nn.Linear`, `nn.LayerNorm`, `torch.einsum`, `torch.sigmoid`, `F.softmax`, `F.relu`. Nothing else.
- **Config-as-object.** A single config object threads hyperparameters (channel dims, number of heads, dropout rates) through every module.
- **Explicit masking throughout.** Every attention and update module accepts optional `seq_mask`, `msa_mask`, or `pair_mask` tensors. Masks propagate from top-level input all the way through to loss computation.
- **nm/Angstrom boundary.** The Structure Module operates internally in nanometres (matching the supplement). The boundary is at `StructureModule.__init__` (converts residue constants from Angstroms to nm) and `StructureModule.forward` (converts outputs back to Angstroms).
- **Zero-init per supplement 1.11.4.** Output projections for attention modules, transition blocks, and head logit layers are zero-initialized. Gate biases are initialized to 1.

## Getting Started

### Requirements

- Python 3.10+
- PyTorch 2.0+
- pytest (for tests)
- `aws` CLI and `unzip` (for downloading OpenProteinSet)

### Install

```bash
pip install torch pytest
```

### Download OpenProteinSet

```bash
python scripts/download_openproteinset.py --data-root data/openproteinset
```

This downloads the minimal raw assets used by the repo and normalizes them into:

```text
data/openproteinset/roda_pdb/<chain_id>/a3m/uniref90_hits.a3m
data/openproteinset/roda_pdb/<chain_id>/hhr/pdb70_hits.hhr
data/openproteinset/pdb_data/mmcif_files/<pdb_id>.cif
```

### Preprocess OpenProteinSet

```bash
python scripts/preprocess_openproteinset.py \
  --raw-root data/openproteinset \
  --processed-features-dir data/processed_features \
  --processed-labels-dir data/processed_labels
```

### Run Tests

```bash
pytest -q
```

The test suite includes 67 tests covering parsers, geometry, dataset processing, and model/loss behavior.

## Work in Progress

### What's done

- Full forward pass: input embedding through Evoformer through Structure Module to all-atom coordinates
- All auxiliary heads: distogram, pLDDT, masked MSA, TM-score, experimentally resolved
- All losses: FAPE (backbone + all-atom), torsion angle, pLDDT, distogram, MSA, structural violations
- Recycling loop with proper gradient detachment and pseudo-beta distance features
- Template processing (pair stack + pointwise attention + torsion angle features)
- Extra MSA stack with global column attention
- Self-contained OpenProteinSet download and preprocessing scripts
- Minimal cached dataset loader with crops, collation, MSA processing, template features, and supervision tensors
- Geometry helpers for frames, torsions, and pseudo-beta coordinates
- Ensemble averaging
- Parameter initialization matching supplement 1.11.4
- 67 parser, shape, semantic, and end-to-end tests

### Next steps

- [ ] Define the training loop
- [ ] Test training on a small set of proteins

## License

MIT. See [LICENSE](LICENSE).

`residue_constants.py` contains data derived from the [AlphaFold2 source code](https://github.com/google-deepmind/alphafold), which is licensed under Apache 2.0.

## Acknowledgments

- Jumper, J. et al. "Highly accurate protein structure prediction with AlphaFold." *Nature* 596, 583-589 (2021). The [supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf) is the primary reference for this implementation.
- Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT) for the inspiration of minimal, readable reimplementations.
