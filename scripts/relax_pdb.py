"""Amber-style energy minimisation of a predicted PDB (supplement 1.8.6).

AlphaFold2 reports that the construction of atom coordinates from independent
backbone frames and torsion angles "produces idealized bond lengths and bond
angles for most of the atom bonds, but the geometry for inter-residue bonds
(peptide bonds) and the avoidance of atom clashes need to be learned." For
pre-fine-tuning checkpoints (our overfit setup) or any model output that has
not had the violation loss active, those inter-residue bonds and clashes are
typically violated — the fold is correct but the chemistry isn't.

The supplement's fix (section 1.8.6) is Amber relaxation: a constrained
energy minimisation with a force field that pulls long peptide bonds back
to ~1.33 A and resolves clashes, while position-restraining heavy atoms to
prevent the fold from drifting. This script is a minimal, pedagogical port
of that procedure.

Usage::

    python scripts/relax_pdb.py path/to/predicted.pdb
    # -> writes path/to/predicted_relaxed.pdb

    python scripts/relax_pdb.py predicted.pdb --output relaxed.pdb --restrain CA

Requires OpenMM and pdbfixer (not installed by the project by default — they
would pull MD-specific C++ extensions that have nothing to do with the model
itself). Install only when you want to relax structures::

    pip install openmm pdbfixer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_INSTALL_HINT = (
    "OpenMM and pdbfixer are required for Amber relaxation. Install with:\n"
    "    pip install openmm pdbfixer\n"
    "Neither is a default project dependency — they pull MD-specific "
    "extensions unrelated to the model."
)


def relax_pdb(
    input_pdb: Path,
    output_pdb: Path,
    restrain_atom_name: str = "CA",
    restraint_k_kcal_per_mol_angstrom_sq: float = 10.0,
    max_iterations: int = 0,
    tolerance_kcal_per_mol: float = 2.39,
) -> dict:
    """Run a CA-restrained Amber energy minimisation on ``input_pdb``.

    Parameters mirror supplement 1.8.6 conventions:

    - ``restrain_atom_name``: atom name (typically ``CA``) whose positions
      are harmonically restrained. Restraining the alpha carbons fixes the
      fold while letting side chains and peptide-bond geometry relax.
    - ``restraint_k``: spring constant on the restrained atoms. 10 kcal/mol/A^2
      is the AF2-release default — stiff enough to preserve the fold within
      ~0.3 A per CA, loose enough to let broken bonds pull straight.
    - ``max_iterations = 0``: minimise until the energy converges below
      ``tolerance``; no iteration cap.
    - ``tolerance = 2.39 kcal/mol``: the OpenMM default (10 kJ/mol in its
      internal units). Tight enough for local geometry to settle.

    Returns a dict with summary stats (energies, max displacement of
    restrained atoms) for logging.
    """
    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
        import pdbfixer
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err

    # ------------------------------------------------------------------
    # 1. Repair the PDB (add missing atoms + hydrogens).
    # Our PDB writer only emits the heavy atoms in atom14 order, but Amber
    # force fields need an explicit-hydrogen model. PDBFixer fills both gaps.
    # ------------------------------------------------------------------
    fixer = pdbfixer.PDBFixer(filename=str(input_pdb))
    fixer.findMissingResidues()
    fixer.missingResidues = {}  # Don't try to rebuild gaps in the chain — we
                                 # want to relax what we have, not hallucinate.
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    # ------------------------------------------------------------------
    # 2. Build the system with Amber99SB + GBSA implicit solvent (AF2 uses
    #    Amber99SB; the OBC implicit solvent keeps the minimisation fast
    #    without needing a water box).
    # ------------------------------------------------------------------
    forcefield = app.ForceField("amber99sb.xml", "amber99_obc.xml")
    system = forcefield.createSystem(
        fixer.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )

    # ------------------------------------------------------------------
    # 3. Add harmonic position restraints on the chosen heavy atoms.
    # ------------------------------------------------------------------
    restraint = openmm.CustomExternalForce(
        "0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
    )
    k = restraint_k_kcal_per_mol_angstrom_sq * unit.kilocalorie_per_mole / unit.angstrom ** 2
    restraint.addGlobalParameter("k", k)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    restrained_indices: list[int] = []
    for atom in fixer.topology.atoms():
        if atom.name == restrain_atom_name:
            pos = fixer.positions[atom.index].value_in_unit(unit.nanometer)
            restraint.addParticle(atom.index, [pos[0], pos[1], pos[2]])
            restrained_indices.append(atom.index)
    if not restrained_indices:
        raise ValueError(
            f"No atoms named {restrain_atom_name!r} found in {input_pdb} — "
            "check the PDB or pass --restrain <name>."
        )
    system.addForce(restraint)

    # ------------------------------------------------------------------
    # 4. Minimise. A zero-step LangevinMiddleIntegrator is just a placeholder
    #    the Simulation API requires; minimizeEnergy does L-BFGS internally.
    # ------------------------------------------------------------------
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    simulation = app.Simulation(fixer.topology, system, integrator)
    simulation.context.setPositions(fixer.positions)

    initial_energy = (
        simulation.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilocalorie_per_mole)
    )
    simulation.minimizeEnergy(
        tolerance=tolerance_kcal_per_mol * unit.kilocalorie_per_mole,
        maxIterations=max_iterations,
    )
    final_energy = (
        simulation.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilocalorie_per_mole)
    )

    # ------------------------------------------------------------------
    # 5. Measure how far the restrained atoms moved (sanity check).
    # ------------------------------------------------------------------
    final_positions = simulation.context.getState(getPositions=True).getPositions()
    max_restrained_drift_angstrom = 0.0
    for atom_index in restrained_indices:
        before = fixer.positions[atom_index].value_in_unit(unit.angstrom)
        after = final_positions[atom_index].value_in_unit(unit.angstrom)
        drift = sum((a - b) ** 2 for a, b in zip(before, after)) ** 0.5
        max_restrained_drift_angstrom = max(max_restrained_drift_angstrom, drift)

    # ------------------------------------------------------------------
    # 6. Write the relaxed PDB.
    # ------------------------------------------------------------------
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb.open("w") as f:
        app.PDBFile.writeFile(fixer.topology, final_positions, f)

    return {
        "input_pdb": str(input_pdb),
        "output_pdb": str(output_pdb),
        "initial_energy_kcal_per_mol": initial_energy,
        "final_energy_kcal_per_mol": final_energy,
        "restrained_atom_name": restrain_atom_name,
        "num_restrained_atoms": len(restrained_indices),
        f"max_{restrain_atom_name.lower()}_drift_angstrom": max_restrained_drift_angstrom,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pdb", type=Path, help="Predicted PDB to relax.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: <input>_relaxed.pdb next to the input).")
    parser.add_argument("--restrain", type=str, default="CA",
                        help="Atom name to position-restrain. Default CA = preserve fold, "
                             "relax side chains + backbone bonds.")
    parser.add_argument("--restraint-k", type=float, default=10.0,
                        help="Restraint spring constant in kcal/mol/A^2 (default 10).")
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="Max minimisation iterations (0 = run to convergence).")
    args = parser.parse_args(argv)

    if not args.input_pdb.is_file():
        print(f"[relax] input PDB not found: {args.input_pdb}", file=sys.stderr)
        sys.exit(1)

    output_pdb = args.output or args.input_pdb.with_name(f"{args.input_pdb.stem}_relaxed.pdb")

    print(f"[relax] input    : {args.input_pdb}")
    print(f"[relax] output   : {output_pdb}")
    print(f"[relax] restrain : {args.restrain} @ k={args.restraint_k} kcal/mol/A^2")

    try:
        stats = relax_pdb(
            args.input_pdb,
            output_pdb,
            restrain_atom_name=args.restrain,
            restraint_k_kcal_per_mol_angstrom_sq=args.restraint_k,
            max_iterations=args.max_iterations,
        )
    except ImportError as err:
        print(f"[relax] {err}", file=sys.stderr)
        sys.exit(2)

    drift_key = f"max_{args.restrain.lower()}_drift_angstrom"
    print(
        f"[relax] energy   : {stats['initial_energy_kcal_per_mol']:.1f} -> "
        f"{stats['final_energy_kcal_per_mol']:.1f} kcal/mol"
    )
    print(
        f"[relax] max {args.restrain} drift: {stats[drift_key]:.3f} A "
        f"(restraint keeps fold near-fixed)"
    )
    print(f"[relax] wrote {output_pdb}")


if __name__ == "__main__":
    main()
