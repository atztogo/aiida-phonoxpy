"""Tests for PhonopyWorkChain."""
import numpy as np
from aiida.orm import Dict, StructureData
from phonopy.structure.cells import isclose

from aiida_phonoxpy.common.utils import phonopy_atoms_from_structure


def test_initialize(generate_workchain, generate_structure, generate_phonopy_settings):
    """Test of PhonopyWorkChain.initialize()."""
    structure = generate_structure()
    settings = generate_phonopy_settings()
    inputs = {"structure": structure, "settings": settings}
    wc = generate_workchain("phonoxpy.phonopy", inputs)
    wc.initialize()

    ctx = {
        "phonon_setting_info": Dict,
        "primitive": StructureData,
        "supercell": StructureData,
        "displacement_dataset": Dict,
        "supercells": dict,
    }

    for key in wc.ctx:
        assert key in ctx
        assert isinstance(wc.ctx[key], ctx[key])

    dataset = {
        "natom": 8,
        "first_atoms": [
            {"number": 0, "displacement": [0.03, 0.0, 0.0]},
            {"number": 4, "displacement": [0.03, 0.0, 0.0]},
        ],
    }
    wc_dataset = wc.ctx.displacement_dataset.get_dict()
    assert wc_dataset["natom"] == dataset["natom"]
    np.testing.assert_almost_equal(
        [d["displacement"] for d in wc_dataset["first_atoms"]],
        [d["displacement"] for d in dataset["first_atoms"]],
    )
    np.testing.assert_equal(
        [d["number"] for d in wc_dataset["first_atoms"]],
        [d["number"] for d in dataset["first_atoms"]],
    )

    num_disps = len(dataset["first_atoms"])
    for i in range(num_disps):
        ndigits = len(str(num_disps))
        num = str(i + 1).zfill(ndigits)
        key = f"supercell_{num}"
        assert key in wc.ctx.supercells
        d = dataset["first_atoms"][i]
        scell_per = phonopy_atoms_from_structure(wc.ctx.supercell)
        pos = scell_per.positions
        pos[d["number"]] += d["displacement"]
        scell_per.positions = pos
        scell_disp = phonopy_atoms_from_structure(wc.ctx.supercells[key])
        assert isclose(scell_disp, scell_per)

    phonon_setting_info_keys = [
        "version",
        "distance",
        "symmetry",
        "primitive_matrix",
        "supercell_matrix",
    ]
    primitive_matrix = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for key in wc.ctx.phonon_setting_info.keys():
        assert key in phonon_setting_info_keys
    np.testing.assert_almost_equal(
        wc.ctx.phonon_setting_info["primitive_matrix"], primitive_matrix
    )
    np.testing.assert_equal(
        wc.ctx.phonon_setting_info["supercell_matrix"], supercell_matrix
    )
