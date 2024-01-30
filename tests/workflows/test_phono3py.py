"""Tests for Phono3pyWorkChain."""

import numpy as np
import pytest
from phonopy.structure.cells import isclose

from aiida_phonoxpy.utils.utils import phonopy_atoms_from_structure


def test_initialize_with_dataset(
    generate_workchain,
    generate_structure,
    generate_displacement_dataset,
    generate_settings,
):
    """Test of Phono3pyWorkChain.initialize() using NaCl data.

    `displacement_dataset` is generated using phono3py.

    """
    structure = generate_structure()
    settings = generate_settings()
    dataset = generate_displacement_dataset(structure_id="NaCl-fc3")
    inputs = {"structure": structure, "settings": settings}
    wc = generate_workchain("phonoxpy.phono3py", inputs)
    wc.initialize()

    phonon_setting_info_keys = [
        "version",
        "distance",
        "symmetry",
        "primitive_matrix",
        "supercell_matrix",
        "symmetry_tolerance",
    ]
    assert "displacement_dataset" not in wc.inputs
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)
    _assert_dataset(wc, dataset)


def test_initialize_with_dataset_with_phonon_supercell_matrix(
    generate_workchain,
    generate_structure,
    generate_displacement_dataset,
    generate_settings,
):
    """Test of Phono3pyWorkChain.initialize() using NaCl data.

    `displacement_dataset` and `phonon_displacement_dataset` are generated
    using phono3py.

    """
    structure = generate_structure()
    settings = generate_settings(phonon_supercell_matrix=[2, 2, 2])
    dataset = generate_displacement_dataset(structure_id="NaCl-fc3")
    phonon_dataset = generate_displacement_dataset(structure_id="NaCl-64")
    inputs = {"structure": structure, "settings": settings}
    wc = generate_workchain("phonoxpy.phono3py", inputs)
    wc.initialize()

    phonon_setting_info_keys = [
        "version",
        "distance",
        "symmetry",
        "primitive_matrix",
        "supercell_matrix",
        "symmetry_tolerance",
        "phonon_supercell_matrix",
    ]
    assert "displacement_dataset" not in wc.inputs
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)

    _assert_dataset(wc, dataset, phonon_dataset=phonon_dataset)


def _assert_dataset(wc, dataset, phonon_dataset=None):
    from aiida.orm import Dict, StructureData

    ctx = {
        "phonon_setting_info": Dict,
        "primitive": StructureData,
        "supercell": StructureData,
        "displacement_dataset": Dict,
        "supercells": dict,
        "phonon_supercell": StructureData,
        "phonon_supercells": dict,
        "phonon_displacement_dataset": Dict,
    }
    for key in wc.ctx:
        assert key in ctx
        assert isinstance(wc.ctx[key], ctx[key])

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
    for wc_first_atoms, first_atoms in zip(
        wc_dataset["first_atoms"], dataset["first_atoms"]
    ):
        np.testing.assert_almost_equal(
            [d["displacement"] for d in wc_first_atoms["second_atoms"]],
            [d["displacement"] for d in first_atoms["second_atoms"]],
        )
        np.testing.assert_equal(
            [d["number"] for d in wc_first_atoms["second_atoms"]],
            [d["number"] for d in first_atoms["second_atoms"]],
        )

    num_first_disps = len(dataset["first_atoms"])
    num_disps = num_first_disps
    for first_atoms in dataset["first_atoms"]:
        num_disps += len(first_atoms["second_atoms"])
    ndigits = len(str(num_disps))
    count = num_first_disps

    for i, first_atoms in enumerate(dataset["first_atoms"]):
        num = str(i + 1).zfill(ndigits)
        key = f"supercell_{num}"
        assert key in wc.ctx.supercells
        d = first_atoms
        scell_per = phonopy_atoms_from_structure(wc.ctx.supercell)
        pos = scell_per.positions
        pos[d["number"]] += d["displacement"]
        scell_per.positions = pos
        scell_disp = phonopy_atoms_from_structure(wc.ctx.supercells[key])
        assert isclose(scell_disp, scell_per)

        for second_atoms in first_atoms["second_atoms"]:
            num = str(count + 1).zfill(ndigits)
            key = f"supercell_{num}"
            assert key in wc.ctx.supercells
            d2 = second_atoms
            scell_per = phonopy_atoms_from_structure(wc.ctx.supercell)
            pos = scell_per.positions
            pos[d["number"]] += d["displacement"]
            pos[d2["number"]] += d2["displacement"]
            scell_per.positions = pos
            scell_disp = phonopy_atoms_from_structure(wc.ctx.supercells[key])
            assert isclose(scell_disp, scell_per)
            count += 1

    if phonon_dataset is not None:
        wc_phonon_dataset = wc.ctx.phonon_displacement_dataset.get_dict()
        assert wc_phonon_dataset["natom"] == phonon_dataset["natom"]
        np.testing.assert_almost_equal(
            [d["displacement"] for d in wc_phonon_dataset["first_atoms"]],
            [d["displacement"] for d in phonon_dataset["first_atoms"]],
        )
        np.testing.assert_equal(
            [d["number"] for d in wc_phonon_dataset["first_atoms"]],
            [d["number"] for d in phonon_dataset["first_atoms"]],
        )
        num_disps = len(phonon_dataset["first_atoms"])
        ndigits = len(str(num_disps))
        for i in range(num_disps):
            num = str(i + 1).zfill(ndigits)
            key = f"phonon_supercell_{num}"
            assert key in wc.ctx.phonon_supercells
            d = phonon_dataset["first_atoms"][i]
            scell_per = phonopy_atoms_from_structure(wc.ctx.phonon_supercell)
            pos = scell_per.positions
            pos[d["number"]] += d["displacement"]
            scell_per.positions = pos
            scell_disp = phonopy_atoms_from_structure(wc.ctx.phonon_supercells[key])
            assert isclose(scell_disp, scell_per)

    _assert_cells(wc)


def _assert_cells(wc):
    primitive_matrix = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    np.testing.assert_almost_equal(
        wc.ctx.phonon_setting_info["primitive_matrix"], primitive_matrix
    )
    np.testing.assert_equal(
        wc.ctx.phonon_setting_info["supercell_matrix"], supercell_matrix
    )
    if "phonon_supercell_matrix" in wc.ctx.phonon_setting_info.keys():
        phonon_supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        np.testing.assert_equal(
            wc.ctx.phonon_setting_info["phonon_supercell_matrix"],
            phonon_supercell_matrix,
        )


@pytest.mark.usefixtures(
    "mock_forces_run_calculation", "mock_nac_params_run_calculation"
)
@pytest.mark.parametrize("plugin_name", ["vasp.vasp", "quantumespresso.pw"])
def test_Phono3pyWorkChain_full(
    generate_structure,
    generate_workchain,
    generate_force_sets,
    generate_nac_params,
    mock_calculator_code,
    plugin_name,
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool, Dict

    force_sets = generate_force_sets(structure_id="NaCl-fc3").get_array("force_sets")
    phonon_force_sets = generate_force_sets(structure_id="NaCl-64").get_array(
        "force_sets"
    )
    nac_params_data = generate_nac_params()
    born_charges = nac_params_data.get_array("born_charges")
    epsilon = nac_params_data.get_array("epsilon")

    settings = {
        "supercell_matrix": [1, 1, 1],
        "phonon_supercell_matrix": [2, 2, 2],
        "distance": 0.03,
    }

    if plugin_name == "vasp.vasp":
        nac_inputs = {
            "code": mock_calculator_code(plugin_name),
            "born_charges": born_charges,
            "epsilon": epsilon,
        }
    elif plugin_name == "quantumespresso.pw":
        nac_inputs = {
            "steps": [
                {"code": mock_calculator_code("quantumespresso.pw")},
                {
                    "code": mock_calculator_code("quantumespresso.ph"),
                },
            ],
            "born_charges": born_charges,
            "epsilon": epsilon,
        }
    else:
        raise RuntimeError("plugin_name doesn't exist.")

    inputs = {
        "structure": generate_structure(),
        "settings": Dict(dict=settings),
        "metadata": {},
        "calculator_inputs": {
            "force": {
                "code": mock_calculator_code(plugin_name),
                "force_sets": force_sets,
            },
            "phonon_force": {
                "code": mock_calculator_code(plugin_name),
                "force_sets": phonon_force_sets,
            },
            "nac": nac_inputs,
        },
        "run_fc": Bool(True),
        "run_ltc": Bool(True),
    }

    process = generate_workchain("phonoxpy.phono3py", inputs)
    results, node = launch.run_get_node(process)

    np.testing.assert_allclose(
        results["force_sets"].get_array("force_sets"),
        force_sets,
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        results["phonon_force_sets"].get_array("force_sets"),
        phonon_force_sets,
        atol=1e-8,
        rtol=0,
    )
    for key in ("born_charges", "epsilon"):
        np.testing.assert_allclose(
            results["nac_params"].get_array(key),
            generate_nac_params().get_array(key),
            atol=1e-8,
            rtol=0,
        )

    output_keys = (
        "displacement_dataset",
        "force_sets",
        "nac_params",
        "phonon_displacement_dataset",
        "phonon_force_sets",
        "phonon_setting_info",
        "phonon_supercell",
        "primitive",
        "supercell",
        "ltc",
        "fc2",
        "fc3",
    )
    assert set(list(results)) == set(output_keys)
