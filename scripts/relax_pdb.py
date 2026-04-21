"""Iterative restrained Amber relaxation — a faithful port of supplement 1.8.6.

Quoting §1.8.6 verbatim (AF2 supplement, page 31):

> "In order to resolve any remaining structural violations and clashes, we
> relax our model predictions by an iterative restrained energy minimization
> procedure. At each round, we perform minimization of the AMBER99SB force
> field with additional harmonic restraints that keep the system near its
> input structure. The restraints are applied independently to heavy atoms,
> with a spring constant of 10 kcal/mol Å². Once the minimizer has converged,
> we determine which residues still contain violations. We then remove
> restraints from all atoms within these residues and perform restrained
> minimization once again, starting from the minimized structure of the
> previous iteration. This process is repeated until all violations are
> resolved."

The violation detection reuses this repo's ``StructuralViolationLoss`` from
``minalphafold.losses``, which implements the paper's supplement §1.9.11
equations 44-47 (bond length τ = 12σ_lit, bond angle τ = 12σ_lit, clash
τ = 1.5 Å). The exact same formulas that drove the training loss now drive
the relaxation loop, so the tools are consistent.

Usage::

    pip install -e '.[relax]'
    python scripts/relax_pdb.py path/to/predicted.pdb

Outputs ``<stem>_relaxed.pdb`` next to the input. Logs per-round: number of
residues still violating, energy, and which residues were freed this round.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

# Ensure the repo root is importable so we can reach minalphafold.* when this
# script is invoked directly (matches the pattern used by the other scripts).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_INSTALL_HINT = (
    "OpenMM and pdbfixer are required for Amber relaxation. Install with:\n"
    "    pip install -e '.[relax]'\n"
    "or via conda-forge for a smoother C++-extension install:\n"
    "    conda install -c conda-forge openmm pdbfixer"
)


def _openmm_positions_to_atom14(topology, positions):
    """Extract atom14-ordered coordinates, masks, aatype, and residue index.

    Matches the ``atom14`` convention used by ``StructuralViolationLoss``:
    14 slots per residue, atom type per slot given by
    ``residue_constants.restype_name_to_atom14_names``, residue index given
    by the 20-AA order (``restype_order``), unknown → 20.

    Hydrogens and non-standard residues are skipped — the violation loss
    operates on heavy atoms only, matching the paper.
    """
    import numpy as np
    from openmm import unit
    from minalphafold.residue_constants import (
        restype_name_to_atom14_names,
        restype_3to1,
        restype_order,
    )

    residues_list = list(topology.residues())
    n_res = len(residues_list)
    atom14_pos = np.zeros((n_res, 14, 3), dtype=np.float32)
    atom_mask = np.zeros((n_res, 14), dtype=np.float32)
    aatype = np.full(n_res, 20, dtype=np.int64)  # UNK default
    residue_index = np.zeros(n_res, dtype=np.int64)

    for i, residue in enumerate(residues_list):
        resname = residue.name
        residue_index[i] = int(residue.id)
        if resname not in restype_3to1:
            continue  # Leave as UNK + all-zero mask; violation loss will skip it.
        aatype[i] = restype_order[restype_3to1[resname]]
        atom14_names = restype_name_to_atom14_names[resname]
        # Build a name → slot lookup that drops the empty-string padding slots.
        name_to_slot = {name: slot for slot, name in enumerate(atom14_names) if name}
        for atom in residue.atoms():
            slot = name_to_slot.get(atom.name)
            if slot is None:
                continue  # Hydrogen or non-standard atom name — not tracked in atom14.
            pos_ang = positions[atom.index].value_in_unit(unit.angstrom)
            atom14_pos[i, slot, :] = [pos_ang[0], pos_ang[1], pos_ang[2]]
            atom_mask[i, slot] = 1.0

    return atom14_pos, atom_mask, aatype, residue_index


def _detect_violating_residues(topology, positions, violation_tolerance_factor=12.0, clash_overlap_tolerance=1.5):
    """Per-residue violation mask per supplement §1.9.11 equations 44-47.

    Reuses ``minalphafold.losses.StructuralViolationLoss`` so the detection
    criteria are bit-identical to the training-time loss. The three masks
    returned by that loss — bond/angle (between-residue), clash
    (between-residue), and within-residue bounds — are OR-combined to a
    single per-residue flag: True iff any violation type fires.
    """
    import numpy as np
    import torch
    from minalphafold.losses import StructuralViolationLoss

    atom14_pos, atom_mask, aatype, residue_index = _openmm_positions_to_atom14(
        topology, positions,
    )
    pos = torch.as_tensor(atom14_pos).unsqueeze(0)      # (1, N_res, 14, 3)
    mask = torch.as_tensor(atom_mask).unsqueeze(0)      # (1, N_res, 14)
    types = torch.as_tensor(aatype).unsqueeze(0)        # (1, N_res)
    resi = torch.as_tensor(residue_index).unsqueeze(0)  # (1, N_res)

    loss = StructuralViolationLoss(
        violation_tolerance_factor=violation_tolerance_factor,
        clash_overlap_tolerance=clash_overlap_tolerance,
    )
    with torch.no_grad():
        bond_angle = loss.between_residue_bond_and_angle_loss(pos, mask, types, resi)
        between_clash = loss.between_residue_clash_loss(pos, mask, types, resi)
        within = loss.within_residue_violation_loss(pos, mask, types)

    # Shapes: each is either (1, N_res) directly or (1, N_res, 14) → reduce on atoms.
    bond_angle_mask = bond_angle["per_residue_violation_mask"][0].bool()
    clash_mask = between_clash["per_atom_clash_mask"][0].any(dim=-1).bool()
    within_mask = within["per_atom_violations"][0].any(dim=-1).bool()

    combined = bond_angle_mask | clash_mask | within_mask
    return combined.cpu().numpy().astype(bool), {
        "n_bond_angle": int(bond_angle_mask.sum().item()),
        "n_between_clash": int(clash_mask.sum().item()),
        "n_within": int(within_mask.sum().item()),
    }


def _soft_start_minimize(simulation, system, tolerance_kj_per_mol_nm: float, max_iterations: int = 100) -> None:
    """Resolve gross initial clashes by scaling Lennard-Jones ε to 1% briefly.

    Paper §1.8.6 does not describe this explicitly, but AF2's reference code
    (``alphafold/relax/amber_minimize.py``) uses the same trick when a
    prediction's starting geometry has severe clashes that would NaN OpenMM's
    L-BFGS at step 0. We do it only as a fallback, triggered by a non-finite
    initial energy.
    """
    import openmm
    import openmm.unit as unit

    nonbonded = None
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, openmm.NonbondedForce):
            nonbonded = force
            break
    if nonbonded is None:
        return

    original_epsilons = []
    for i in range(nonbonded.getNumParticles()):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        original_epsilons.append(epsilon)
        nonbonded.setParticleParameters(i, charge, sigma, epsilon * 0.01)
    nonbonded.updateParametersInContext(simulation.context)

    simulation.minimizeEnergy(
        tolerance=tolerance_kj_per_mol_nm * 10.0 * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=max_iterations,
    )

    for i, epsilon in enumerate(original_epsilons):
        charge, sigma, _ = nonbonded.getParticleParameters(i)
        nonbonded.setParticleParameters(i, charge, sigma, epsilon)
    nonbonded.updateParametersInContext(simulation.context)


def relax_pdb(
    input_pdb: Path,
    output_pdb: Path,
    restraint_k_kcal_per_mol_angstrom_sq: float = 10.0,
    max_rounds: int = 10,
    max_iterations_per_round: int = 0,
    force_tolerance_kj_per_mol_nm: float = 10.0,
    violation_tolerance_factor: float = 12.0,
    clash_overlap_tolerance: float = 1.5,
) -> dict:
    """Iterative restrained Amber relaxation per supplement §1.8.6.

    Parameters match the paper:

    - ``restraint_k = 10 kcal/mol/Å²`` per §1.8.6.
    - Restraints applied to **every heavy atom** (not just Cα).
    - ``violation_tolerance_factor = 12`` and ``clash_overlap_tolerance = 1.5 Å``
      per supplement §1.9.11 — the detection criteria that drove training also
      drive the relaxation loop.
    - ``max_iterations_per_round = 0`` (unbounded per-round, OpenMM default).
    - ``force_tolerance = 10 kJ/mol/nm`` (equivalent to the paper's
      ``2.39 kcal/mol`` tolerance under OpenMM 8's force-unit API).

    Returns a dict with per-round stats.
    """
    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
        import pdbfixer
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err

    # ------------------------------------------------------------------
    # 1. Fix missing heavy atoms (PDBFixer). Leave gaps in the chain alone.
    # ------------------------------------------------------------------
    fixer = pdbfixer.PDBFixer(filename=str(input_pdb))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # ------------------------------------------------------------------
    # 2. Add hydrogens. Modeller's internal minimise step fails ("infinite
    # or NaN") on severely degenerate inputs; we catch and explain.
    # ------------------------------------------------------------------
    modeller = app.Modeller(fixer.topology, fixer.positions)
    try:
        modeller.addHydrogens(pH=7.0)
    except openmm.OpenMMException as err:
        if "infinite or NaN" in str(err):
            raise RuntimeError(
                f"Cannot place hydrogens on {input_pdb}: heavy-atom geometry "
                "has overlapping atoms (infinite Lennard-Jones at step 0). "
                "This script's H placement assumes a plausible heavy-atom "
                "skeleton; an early-training checkpoint with near-identity "
                "rigid frames (same-type residues at identical coords) trips "
                "this. Converged predictions — fold right, peptide bonds "
                "mildly broken — work fine."
            ) from err
        raise
    topology = modeller.topology

    # ------------------------------------------------------------------
    # 3. Build the Amber99SB + GBSA (OBC) implicit-solvent system.
    # `constraints=None` on purpose: SHAKE-style H-bond constraints can fail
    # on distorted initial geometry; we run unconstrained minimisation.
    # ------------------------------------------------------------------
    forcefield = app.ForceField("amber99sb.xml", "amber99_obc.xml")
    system = forcefield.createSystem(
        topology, nonbondedMethod=app.NoCutoff, constraints=None,
    )

    # ------------------------------------------------------------------
    # 4. Per-particle harmonic restraints on every heavy atom.
    # §1.8.6: "The restraints are applied independently to heavy atoms,
    # with a spring constant of 10 kcal/mol Å²."
    #
    # ``k_active`` is per-particle so we can zero it on specific atoms
    # between rounds (freeing residues whose violations haven't resolved)
    # without rebuilding the system.
    # ------------------------------------------------------------------
    restraint = openmm.CustomExternalForce(
        "0.5 * k_active * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
    )
    restraint.addPerParticleParameter("k_active")
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    k_full = (
        restraint_k_kcal_per_mol_angstrom_sq
        * unit.kilocalorie_per_mole / unit.angstrom ** 2
    ).value_in_unit(unit.kilojoule_per_mole / unit.nanometer ** 2)

    # ``restraint_particle_idx_per_atom[atom.index]`` = particle index inside
    # the restraint force (not the same as atom.index in the system).
    restraint_particle_idx_per_atom: dict[int, int] = {}
    heavy_atoms_per_residue: dict[int, list[int]] = defaultdict(list)
    residues_in_topology: list = list(topology.residues())
    # Map topology.residue.index → sequential 0..N_res-1 index used by
    # _openmm_positions_to_atom14 (they're identical given topology iteration
    # order, but spell it out explicitly).
    residue_seq_index: dict[int, int] = {
        r.index: i for i, r in enumerate(residues_in_topology)
    }

    for atom in topology.atoms():
        if atom.element is not None and atom.element.symbol == "H":
            continue
        pos_nm = modeller.positions[atom.index].value_in_unit(unit.nanometer)
        idx = restraint.addParticle(
            atom.index, [k_full, pos_nm[0], pos_nm[1], pos_nm[2]]
        )
        restraint_particle_idx_per_atom[atom.index] = idx
        heavy_atoms_per_residue[residue_seq_index[atom.residue.index]].append(atom.index)
    system.addForce(restraint)

    # ------------------------------------------------------------------
    # 5. Set up the simulation and minimise iteratively (§1.8.6 loop).
    # ------------------------------------------------------------------
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    force_tolerance = (
        force_tolerance_kj_per_mol_nm * unit.kilojoule_per_mole / unit.nanometer
    )

    def _current_energy() -> float:
        return (
            simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilocalorie_per_mole)
        )

    initial_energy = _current_energy()
    used_soft_start = False
    if not math.isfinite(initial_energy):
        print("[relax] initial energy non-finite; soft-starting (LJ ε × 0.01).")
        _soft_start_minimize(
            simulation, system,
            tolerance_kj_per_mol_nm=force_tolerance_kj_per_mol_nm,
        )
        used_soft_start = True
        initial_energy = _current_energy()

    unrestrained_residues: set[int] = set()
    round_stats: list[dict] = []
    rounds_run = 0
    final_energy = initial_energy
    converged = False

    for round_idx in range(1, max_rounds + 1):
        # Update restraint parameters for every heavy atom:
        #   - k_active = 0 for atoms in unrestrained residues
        #   - targets = current positions (so still-restrained atoms are
        #     pinned to where the previous round's minimise left them)
        current_state = simulation.context.getState(getPositions=True)
        current_positions = current_state.getPositions()
        for res_seq_idx, atom_indices in heavy_atoms_per_residue.items():
            k = 0.0 if res_seq_idx in unrestrained_residues else k_full
            for atom_idx in atom_indices:
                param_idx = restraint_particle_idx_per_atom[atom_idx]
                pos_nm = current_positions[atom_idx].value_in_unit(unit.nanometer)
                restraint.setParticleParameters(
                    param_idx, atom_idx, [k, pos_nm[0], pos_nm[1], pos_nm[2]]
                )
        restraint.updateParametersInContext(simulation.context)

        # Minimise.
        try:
            simulation.minimizeEnergy(
                tolerance=force_tolerance, maxIterations=max_iterations_per_round,
            )
        except openmm.OpenMMException as err:
            if "infinite or NaN" in str(err) and not used_soft_start:
                print(f"[relax] round {round_idx}: minimise NaN'd; retrying with soft-start.")
                _soft_start_minimize(
                    simulation, system,
                    tolerance_kj_per_mol_nm=force_tolerance_kj_per_mol_nm,
                )
                used_soft_start = True
                simulation.minimizeEnergy(
                    tolerance=force_tolerance, maxIterations=max_iterations_per_round,
                )
            else:
                raise

        energy_after = _current_energy()

        # Detect violations (§1.9.11 eqs 44-47) on the relaxed heavy atoms.
        relaxed_positions = simulation.context.getState(getPositions=True).getPositions()
        violation_mask, breakdown = _detect_violating_residues(
            topology, relaxed_positions,
            violation_tolerance_factor=violation_tolerance_factor,
            clash_overlap_tolerance=clash_overlap_tolerance,
        )
        n_violating = int(violation_mask.sum())

        # Residues *newly* violating this round — the ones we'll free before
        # the next round.
        newly_unrestrained = [
            i for i, v in enumerate(violation_mask)
            if v and i not in unrestrained_residues
        ]
        round_stats.append({
            "round": round_idx,
            "energy_kcal_per_mol": energy_after,
            "n_violating_residues": n_violating,
            "n_bond_angle_violations": breakdown["n_bond_angle"],
            "n_between_clash_violations": breakdown["n_between_clash"],
            "n_within_violations": breakdown["n_within"],
            "n_currently_unrestrained": len(unrestrained_residues),
            "n_freed_this_round": len(newly_unrestrained),
        })
        rounds_run = round_idx
        final_energy = energy_after

        print(
            f"[relax] round {round_idx}: energy={energy_after:.1f} kcal/mol, "
            f"{n_violating} residues violating "
            f"(bond/angle={breakdown['n_bond_angle']}, "
            f"between-clash={breakdown['n_between_clash']}, "
            f"within={breakdown['n_within']}); "
            f"freeing {len(newly_unrestrained)} residue(s) for next round"
        )

        if n_violating == 0:
            converged = True
            break
        if not newly_unrestrained:
            # Every violating residue was already unrestrained — we can't
            # make progress by freeing more. Stop.
            print(
                f"[relax] round {round_idx}: {n_violating} residue(s) still "
                "violating but all already unrestrained; stopping."
            )
            break
        unrestrained_residues.update(newly_unrestrained)

    if not converged:
        print(
            f"[relax] WARNING: did not reach zero violations in {rounds_run} "
            "round(s). Output is still written; inspect round_stats for details."
        )

    # ------------------------------------------------------------------
    # 6. Measure drift from the input, split by backbone vs all heavy and
    # by restrained vs unrestrained so the caller can tell whether the
    # fold was preserved (backbone-restrained drift should be ~sub-Å).
    # ------------------------------------------------------------------
    final_positions = simulation.context.getState(getPositions=True).getPositions()
    max_backbone_drift_angstrom = 0.0
    max_restrained_heavy_drift_angstrom = 0.0
    max_any_heavy_drift_angstrom = 0.0
    for atom in topology.atoms():
        if atom.element is not None and atom.element.symbol == "H":
            continue
        before = modeller.positions[atom.index].value_in_unit(unit.angstrom)
        after = final_positions[atom.index].value_in_unit(unit.angstrom)
        drift = sum((a - b) ** 2 for a, b in zip(before, after)) ** 0.5
        max_any_heavy_drift_angstrom = max(max_any_heavy_drift_angstrom, drift)
        if atom.name in ("N", "CA", "C"):
            max_backbone_drift_angstrom = max(max_backbone_drift_angstrom, drift)
        res_seq_idx = residue_seq_index[atom.residue.index]
        if res_seq_idx not in unrestrained_residues:
            max_restrained_heavy_drift_angstrom = max(max_restrained_heavy_drift_angstrom, drift)

    # ------------------------------------------------------------------
    # 7. Write the relaxed PDB.
    # ------------------------------------------------------------------
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb.open("w") as f:
        app.PDBFile.writeFile(topology, final_positions, f)

    return {
        "input_pdb": str(input_pdb),
        "output_pdb": str(output_pdb),
        "initial_energy_kcal_per_mol": initial_energy,
        "final_energy_kcal_per_mol": final_energy,
        "used_soft_start": used_soft_start,
        "rounds_run": rounds_run,
        "converged": converged,
        "max_backbone_drift_angstrom": max_backbone_drift_angstrom,
        "max_restrained_heavy_drift_angstrom": max_restrained_heavy_drift_angstrom,
        "max_any_heavy_drift_angstrom": max_any_heavy_drift_angstrom,
        "n_residues_unrestrained_at_end": len(unrestrained_residues),
        "round_stats": round_stats,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pdb", type=Path, help="Predicted PDB to relax.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: <input>_relaxed.pdb next to the input).")
    parser.add_argument("--restraint-k", type=float, default=10.0,
                        help="Heavy-atom restraint spring constant in kcal/mol/Å² "
                             "(supplement §1.8.6: 10).")
    parser.add_argument("--max-rounds", type=int, default=10,
                        help="Maximum number of iterative relaxation rounds.")
    parser.add_argument("--max-iterations-per-round", type=int, default=0,
                        help="Per-round L-BFGS iteration cap (0 = unbounded, "
                             "supplement §1.8.6 default).")
    parser.add_argument("--force-tolerance", type=float, default=10.0,
                        help="Force-tolerance for convergence (kJ/mol/nm); "
                             "OpenMM default 10 ≈ supplement's 2.39 kcal/mol.")
    parser.add_argument("--violation-tolerance-factor", type=float, default=12.0,
                        help="Bond length/angle tolerance in units of σ_lit "
                             "(supplement §1.9.11).")
    parser.add_argument("--clash-overlap-tolerance", type=float, default=1.5,
                        help="Clash tolerance τ in Å (supplement §1.9.11 eq 46).")
    args = parser.parse_args(argv)

    if not args.input_pdb.is_file():
        print(f"[relax] input PDB not found: {args.input_pdb}", file=sys.stderr)
        sys.exit(1)

    output_pdb = args.output or args.input_pdb.with_name(f"{args.input_pdb.stem}_relaxed.pdb")

    print(f"[relax] input       : {args.input_pdb}")
    print(f"[relax] output      : {output_pdb}")
    print(f"[relax] restraint k : {args.restraint_k} kcal/mol/Å² on all heavy atoms")
    print(f"[relax] max rounds  : {args.max_rounds}")

    try:
        stats = relax_pdb(
            args.input_pdb,
            output_pdb,
            restraint_k_kcal_per_mol_angstrom_sq=args.restraint_k,
            max_rounds=args.max_rounds,
            max_iterations_per_round=args.max_iterations_per_round,
            force_tolerance_kj_per_mol_nm=args.force_tolerance,
            violation_tolerance_factor=args.violation_tolerance_factor,
            clash_overlap_tolerance=args.clash_overlap_tolerance,
        )
    except ImportError as err:
        print(f"[relax] {err}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as err:
        print(f"[relax] {err}", file=sys.stderr)
        sys.exit(3)

    print(
        f"[relax] converged   : {stats['converged']} "
        f"({stats['rounds_run']} round(s))"
    )
    print(
        f"[relax] energy      : {stats['initial_energy_kcal_per_mol']:.1f} -> "
        f"{stats['final_energy_kcal_per_mol']:.1f} kcal/mol"
        + ("  (after soft-start)" if stats["used_soft_start"] else "")
    )
    print(
        f"[relax] drift      : backbone={stats['max_backbone_drift_angstrom']:.3f} Å, "
        f"restrained-heavy={stats['max_restrained_heavy_drift_angstrom']:.3f} Å, "
        f"any-heavy={stats['max_any_heavy_drift_angstrom']:.3f} Å"
    )
    print(
        f"[relax] freed      : {stats['n_residues_unrestrained_at_end']} residue(s) "
        "(unrestrained by end of relaxation)"
    )
    print(f"[relax] wrote {output_pdb}")


if __name__ == "__main__":
    main()
