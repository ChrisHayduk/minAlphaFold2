import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "minalphafold"))


from a3m import sequence_to_ids
from mmcif import extract_chain_atoms


def test_extract_chain_atoms_uses_label_chain_fallback_and_altloc_priority(tmp_path):
    mmcif_path = tmp_path / "toy.cif"
    mmcif_path.write_text(
        "data_toy\n"
        "_entity_poly.entity_id 1\n"
        "_entity_poly.pdbx_seq_one_letter_code_can AG\n"
        "_refine.ls_d_res_high 1.25\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.pdbx_PDB_model_num\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.auth_atom_id\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "ATOM 1 X A 1 1 . ALA ALA N N 0.0 0.0 0.0 1.0\n"
        "ATOM 1 X A 1 1 . ALA ALA CA CA 1.0 0.0 0.0 1.0\n"
        "ATOM 1 X A 1 1 . ALA ALA C C 2.0 0.1 0.0 1.0\n"
        "ATOM 1 X A 1 1 . ALA ALA O O 2.7 0.5 0.0 1.0\n"
        "ATOM 1 X A 1 1 B ALA ALA CB CB 9.0 9.0 9.0 1.0\n"
        "ATOM 1 X A 1 1 A ALA ALA CB CB 1.0 2.0 3.0 0.5\n"
        "ATOM 1 X A 1 2 . GLY GLY N N 3.5 0.0 0.0 1.0\n"
        "ATOM 1 X A 1 2 . GLY GLY CA CA 4.5 0.0 0.0 1.0\n"
        "ATOM 1 X A 1 2 . GLY GLY C C 5.5 0.1 0.0 1.0\n"
        "#\n"
    )

    chain = extract_chain_atoms(mmcif_path, "1abc", "A")

    assert chain.pdb_id == "1abc"
    assert chain.chain_id == "A"
    assert chain.sequence == "AG"
    assert chain.aatype.tolist() == sequence_to_ids("AG").tolist()
    assert chain.residue_index.tolist() == [0, 1]
    assert chain.resolution == 1.25
    assert chain.atom14_positions.shape == (2, 14, 3)
    assert chain.atom14_mask.shape == (2, 14)
    assert np.allclose(chain.atom14_positions[0, 4], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert chain.atom14_mask[1, 4] == 0.0


def test_extract_chain_atoms_defaults_resolution_to_zero_when_missing(tmp_path):
    mmcif_path = tmp_path / "toy_missing_resolution.cif"
    mmcif_path.write_text(
        "data_toy\n"
        "_entity_poly.entity_id 1\n"
        "_entity_poly.pdbx_seq_one_letter_code_can A\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.pdbx_PDB_model_num\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.auth_atom_id\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "ATOM 1 A A 1 1 . ALA ALA N N 0.0 0.0 0.0 1.0\n"
        "ATOM 1 A A 1 1 . ALA ALA CA CA 1.0 0.0 0.0 1.0\n"
        "ATOM 1 A A 1 1 . ALA ALA C C 2.0 0.0 0.0 1.0\n"
        "#\n"
    )

    chain = extract_chain_atoms(mmcif_path, "1abc", "A")

    assert chain.resolution == 0.0
    assert chain.residue_index.tolist() == [0]


def test_extract_chain_atoms_uses_canonical_sequence_indices_not_author_numbering(tmp_path):
    mmcif_path = tmp_path / "toy_author_numbering.cif"
    mmcif_path.write_text(
        "data_toy\n"
        "_entity_poly.entity_id 1\n"
        "_entity_poly.pdbx_seq_one_letter_code_can AGA\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.pdbx_PDB_model_num\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.auth_atom_id\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "ATOM 1 A A 1 1 . 10 ALA ALA N N 0.0 0.0 0.0 1.0\n"
        "ATOM 1 A A 1 1 . 10 ALA ALA CA CA 1.0 0.0 0.0 1.0\n"
        "ATOM 1 A A 1 1 . 10 ALA ALA C C 2.0 0.0 0.0 1.0\n"
        "ATOM 1 A A 1 2 . 11 GLY GLY N N 3.0 0.0 0.0 1.0\n"
        "ATOM 1 A A 1 2 . 11 GLY GLY CA CA 4.0 0.0 0.0 1.0\n"
        "ATOM 1 A A 1 2 . 11 GLY GLY C C 5.0 0.0 0.0 1.0\n"
        "ATOM 1 A A 1 3 . 13 ALA ALA N N 6.0 0.0 0.0 1.0\n"
        "ATOM 1 A A 1 3 . 13 ALA ALA CA CA 7.0 0.0 0.0 1.0\n"
        "ATOM 1 A A 1 3 . 13 ALA ALA C C 8.0 0.0 0.0 1.0\n"
        "#\n"
    )

    chain = extract_chain_atoms(mmcif_path, "1abc", "A")

    assert chain.sequence == "AGA"
    assert chain.residue_index.tolist() == [0, 1, 2]
