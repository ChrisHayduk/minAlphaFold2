# minAlphaFold2

A minimal, pedagogical PyTorch reimplementation of [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2).

Inspired by Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT).

<p align="center">
  <video src="https://github.com/user-attachments/assets/7bb75c73-1843-452e-9193-ebcb596471f1" width="720" autoplay loop muted playsinline controls></video>
</p>
<p align="center">
  <em>minAlphaFold2 prediction (cyan) overlaid with the 6M0J crystal ground truth (magenta) and DeepMind's AlphaFold2 (orange).</em>
</p>

## Philosophy

- **Pure PyTorch.** Every layer is built from `nn.Linear`, `nn.LayerNorm`, `torch.einsum`, and standard activations. No external ML libraries, no hidden abstractions.
- **1-to-1 mapping to the supplement.** Each module corresponds directly to numbered algorithms in the [AlphaFold2 supplementary information](https://www.nature.com/articles/s41586-021-03819-2#Sec20). Comments reference specific algorithm and line numbers throughout.
- **Designed to be read and modified.** The entire model fits in ~4,000 lines across 15 modules. If you can read PyTorch, you can read this.

## File Structure

```
minalphafold/
    a3m.py                # A3M parsing and MSA tokenization
    mmcif.py              # mmCIF atom-site parsing → atom14 coordinates
    pdbio.py              # PDB writer for predicted structures (pLDDT → B-factor)
    geometry.py           # Rigid frames, torsions, pseudo-β helpers for supervision
    residue_constants.py  # Amino acid chemical data (frames, bond lengths, VDW, torsion masks)
    data.py               # Processed-cache dataset, crops, collation, feature builders
    initialization.py     # Linear init helpers (default/relu/glorot/final, gate, zero)
    utils.py              # Row/column dropout, distance binning, recycling distogram
    embedders.py          # Input embedding, RelPos, every attention/update submodule (Alg 8–19)
    evoformer.py          # Evoformer block (Alg 6); MSA row attention with pair bias (Alg 7)
    structure_module.py   # Structure Module, IPA, backbone update, all-atom coordinates
    heads.py              # Distogram, pLDDT, masked MSA, PAE/TM-score, experimentally-resolved
    losses.py             # FAPE (backbone + all-atom), torsion, pLDDT, distogram, MSA, violations
    model.py              # Top-level AlphaFold2, recycling loop, ensemble averaging
    model_config.py       # Typed ModelConfig dataclass — schema for configs/*.toml
    trainer.py            # Training loop, dataloader wiring, checkpoint helpers, load_model_config
configs/
    tiny.toml                    # Shrunk-to-CPU profile (default for tests / smoke runs)
    medium.toml                  # Mid-sized profile for local overfit experiments
    alphafold2.toml              # Paper-spec monomer config (supplement 1.5–1.8 exact)
scripts/
    download_openproteinset.py   # OpenProteinSet downloader
    preprocess_openproteinset.py # Raw OpenProteinSet → per-chain NPZ caches
    overfit_single_pdb.py        # Self-contained single-PDB overfit driver (no MSAs/templates)
    overfit_processed_chain.py   # Full-pipeline overfit on one preprocessed chain
    modal_overfit.py             # Modal Labs GPU wrapper for overfit_processed_chain
    modal_overfit_single_pdb.py  # Modal Labs GPU wrapper for overfit_single_pdb
    relax_pdb.py                 # Amber-style structure relaxation (supplement 1.8.6)
tests/
    conftest.py                  # Adds repo root + scripts/ to sys.path for pytest
    test_shapes.py               # Shape + semantic tests for every module
    test_a3m.py                  # A3M parser tests
    test_mmcif.py                # mmCIF parser tests
    test_pdbio.py                # PDB writer tests
    test_geometry.py             # Geometry helper tests
    test_data_pipeline.py        # Dataset, preprocessing, end-to-end batch tests
    test_losses.py               # Loss head tests (FAPE, violations, torsion, pLDDT, …)
    test_trainer.py              # Training-loop + optimiser + CLI tests
    test_openproteinset_scripts.py  # Download/preprocess script tests
af2_paper.pdf                    # AF2 supplement — PRIMARY REFERENCE
```

## Supplement Algorithm Mapping

<p align="center">
  <img src="assets/af2_img.png" alt="AlphaFold 2 architecture diagram" width="600">
  <br>
  <em>Diagram of AlphaFold 2 as published in DeepMind's blogpost in November 2020.</em>
</p>

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
| 25 | Rigid-group Frames from Torsions | `structure_module.py: make_rot_x`, `compose_transforms`, `rigid_group_frames_from_torsions` |
| 26 | Rename Symmetric Ground Truth Atoms | `losses.py: select_best_atom14_ground_truth`; ground-truth side: `data.py: build_supervision` (computes `true_torsion_angles_alt` via `geometry.alternative_torsion_angles`) |
| 27 | Torsion Angle Loss | `losses.py: TorsionAngleLoss` |
| 28 | FAPE (Backbone) | `losses.py: BackboneFAPE` |
| 28 | FAPE (All-Atom) | `losses.py: AllAtomFAPE` |
| 29 | PLDDT Head | `heads.py: PLDDTHead` & `losses.py: PLDDTLoss` |
| 30 | Inference with Recycling | `model.py: AlphaFold2.forward` (fixed number of cycles during inference) |
| 31 | Training with Recycling | `model.py: AlphaFold2.forward` (random cycle sampling) |
| 32 | Recycling Embedder | `model.py: AlphaFold2.forward` (recycle norms + distance bins) |

Loss terms beyond the algorithm table: `losses.StructuralViolationLoss` implements supplement §1.9.11 equations 44–47 (bond length, bond angle, clash), `losses.DistogramLoss` implements §1.9.8 eq 41, `losses.MSALoss` implements §1.9.9 eq 42, `losses.ExperimentallyResolvedLoss` implements §1.9.10 eq 43, `losses.TMScoreLoss` implements §1.9.7 eqs 38–40.

## Key Design Decisions

- **Pure PyTorch primitives.** `nn.Linear`, `nn.LayerNorm`, `torch.einsum`, `torch.sigmoid`, `F.softmax`, `F.relu`. Nothing else.
- **Config-as-object.** A single config object threads hyperparameters (channel dims, number of heads, dropout rates) through every module. Channel dim conventions: `c_m` (MSA), `c_s` (single), `c_z` (pair), `c_e` (extra MSA), `c_t` (template pair). Projection `c_m → c_s` happens via `single_rep_proj` in `model.py`.
- **Explicit masking throughout.** Every attention and update module accepts optional `seq_mask`, `msa_mask`, or `pair_mask` tensors. Masks propagate from top-level input all the way through to loss computation.
- **nm/Å boundary at the Structure Module edge.** The Structure Module operates internally in nanometres (matching the supplement). The boundary is at `StructureModule.__init__` (converts residue constants from Å to nm) and `StructureModule.forward` (converts outputs back to Å). No unit mixing inside.
- **Zero-init per supplement 1.11.4.** Output projections for attention modules, transition blocks, and head logit layers are zero-initialized. Gate biases are initialized to 1 (`sigmoid(1) ≈ 0.73`, mostly pass-through). `AlphaFold2._initialize_alphafold_parameters` enforces this sweep after construction.
- **Relative imports inside the package.** Every `minalphafold/*.py` uses `from .X import Y` — no dual-path try/except shims. Tests and scripts put the repo root on `sys.path` (via `tests/conftest.py` or a one-liner at the top of each script) and then use `from minalphafold.X import Y`.

## Getting Started

### Requirements

- Python 3.11+ (the trainer parses TOML config profiles with stdlib `tomllib`).
- `aws` CLI and `unzip`, only if you plan to download OpenProteinSet.

### Install

The project ships a `pyproject.toml` with a minimal core (`torch`, `numpy`) and three opt-in extras so you can install only what you need:

```bash
git clone <this repo>
cd min-AlphaFold

pip install -e .                  # core: torch, numpy
pip install -e '.[dev]'           # + pytest (to run the test suite)
pip install -e '.[relax]'         # + openmm, pdbfixer (Amber relaxation)
pip install -e '.[modal]'         # + modal (cloud-GPU runners in scripts/modal_*.py)
pip install -e '.[all]'           # everything at once
```

Editable install (`-e`) keeps the source tree live — edits to `minalphafold/*.py` take effect without reinstalling. The test suite's `conftest.py` and the scripts' short `sys.path` preamble also put the repo root on the import path automatically, so most workflows don't strictly *need* `pip install`. The pyproject is mostly there so `pip` can resolve dependencies for you and give you clean extras groups.

OpenMM is easier to install via conda if pip gives trouble (C++ extensions):

```bash
conda install -c conda-forge openmm pdbfixer
```

### Overfit a single protein (no MSA, no templates)

The fastest way to verify the pipeline runs end-to-end — parses a PDB,
builds all input features from just the sequence, trains to
sub-Å RMSD on CPU in about a minute:

```bash
python scripts/overfit_single_pdb.py \
  --pdb artifacts/overfit_1a0m_A/ground_truth_1a0m_A.pdb \
  --steps 1000
```

Artifacts (predicted PDB, ground-truth PDB, PyMOL view script, per-step loss log) land in `artifacts/overfit_single_pdb/<chain_id>/`.

### Relax a predicted structure (supplement 1.8.6)

Per §1.9.11 of the supplement:

> *"The construction of the atom coordinates from independent backbone frames and torsion-angles produces idealized bond lengths and bond angles for most of the atom bonds, but the geometry for inter-residue bonds (peptide bonds) and the avoidance of atom clashes need to be learned."*

In other words, a converged *pre-fine-tuning* prediction typically has the right fold but individually broken peptide bonds and occasional clashes — FAPE is frame-invariant, so it never directly constrains `|C_i → N_{i+1}| ≈ 1.33 Å`. Section 1.8.6 describes the paper's fix: an **iterative restrained energy minimization** with Amber99SB and per-heavy-atom position restraints. `scripts/relax_pdb.py` is a faithful port of that procedure.

Each round, per §1.8.6:

1. Minimize the AMBER99SB + GBSA (OBC) implicit-solvent system with a harmonic restraint (`k = 10 kcal/mol/Å²`) on **every heavy atom**, target positions = current positions.
2. Detect residues still containing violations using the exact training-time criteria from supplement §1.9.11 eqs 44-47 — bond-length ± 12σ, bond-angle cos ± 12σ, clash τ = 1.5 Å. Detection reuses `minalphafold.losses.StructuralViolationLoss` so the rules are bit-identical to the training loss.
3. **Remove restraints on all atoms within violating residues**; start the next round from the round's minimized structure.
4. Stop when no residues violate (or no further progress — every violating residue already unrestrained).

```bash
pip install -e '.[relax]'
python scripts/relax_pdb.py artifacts/overfit_processed_chain/6m0j_E/predicted_6m0j_E.pdb
# writes predicted_6m0j_E_relaxed.pdb next to the input
```

Per-round output lists how many residues violate each rule (bond/angle, between-residue clash, within-residue bounds), how many were freed for the next round, and the current energy. Final output includes three drift metrics — backbone-only, restrained-heavy, and any-heavy — so you can tell whether the fold was preserved (backbone/restrained-heavy should be sub-Å on realistic predictions).

Caveat, also acknowledged by the paper: this procedure is designed for mildly-violating inputs. §1.8.6 ends with *"In the CASP14 assessment we used a single iteration; targets with unresolved violations were re-run"*. A pre-fine-tuning overfit checkpoint can have 30-40% of residues violating — too many for this loop to resolve without freeing so much of the chain that neighboring bond forces drag the restrained regions with them. The paper's "re-run" escape hatch isn't available here; if you need cleaner chemistry, train with the violation loss enabled (see `--violations-after-step` on `scripts/overfit_processed_chain.py`).

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

### Train

```bash
python -m minalphafold.trainer \
  --processed-features-dir data/processed_features \
  --processed-labels-dir data/processed_labels \
  --epochs 1 \
  --batch-size 1
```

### Run Tests

```bash
pytest -q
```

The test suite has **130 tests** across parsers (`test_a3m`, `test_mmcif`, `test_pdbio` — 8), geometry (`test_geometry` — 12), dataset and preprocessing (`test_data_pipeline`, `test_openproteinset_scripts` — 26), loss heads (`test_losses` — 15), shape and semantic coverage of every model module (`test_shapes` — 54), and training-loop behaviour (`test_trainer` — 15).

## Work in Progress

### What's done

- Full forward pass: input embedding → Evoformer → Structure Module → all-atom coordinates
- All auxiliary heads: distogram, pLDDT, masked MSA, PAE/pTM, experimentally resolved
- All losses: FAPE (backbone + all-atom), torsion angle, pLDDT, distogram, MSA, structural violations
- Recycling loop with proper gradient detachment and pseudo-β distance features (Algorithm 32)
- Template processing (pair stack + pointwise attention + torsion angle features)
- Extra MSA stack with global column attention
- Self-contained OpenProteinSet download and preprocessing scripts
- Cached dataset loader with crops, collation, MSA processing, template features, and supervision tensors
- Minimal training loop with data → model → loss wiring and checkpoints
- Geometry helpers for frames, torsions, and pseudo-β coordinates
- Ensemble averaging
- Parameter initialization matching supplement 1.11.4 (centralised sweep in `AlphaFold2._initialize_alphafold_parameters`)
- Single-protein overfit driver (`scripts/overfit_single_pdb.py`) reaching sub-Å Cα RMSD in ≤1000 CPU steps
- Iterative restrained Amber relaxation (`scripts/relax_pdb.py`, supplement 1.8.6) — OpenMM + Amber99SB, all heavy atoms restrained, violation detection reuses `StructuralViolationLoss` (§1.9.11 eqs 44-47)
- Gradient checkpointing / rematerialization on the Evoformer + Extra MSA stacks (supplement 1.11.8) — lets the paper-spec 48-block Evoformer fit in GPU memory at full-chain crop sizes
- Modal Labs cloud-GPU runners (`scripts/modal_overfit*.py`) for full-scale training
- 130 parser, shape, semantic, loss, and end-to-end tests

### Next steps

- [ ] Train on a small set of proteins and iterate on default hyperparameters
- [ ] Self-distillation dataset generation (supplement 1.3)

## License

MIT. See [LICENSE](LICENSE).

`residue_constants.py` contains data derived from the [AlphaFold2 source code](https://github.com/google-deepmind/alphafold), which is licensed under Apache 2.0.

## Acknowledgments

- Jumper, J. et al. "Highly accurate protein structure prediction with AlphaFold." *Nature* 596, 583-589 (2021). The [supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf) is the primary reference for this implementation.
- Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT) for the inspiration of minimal, readable reimplementations.
