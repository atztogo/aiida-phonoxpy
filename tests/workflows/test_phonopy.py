"""Tests for PhonopyWorkChain."""

import numpy as np
import pytest
import tempfile
import h5py
import shutil
from phonopy.structure.cells import isclose

from aiida_phonoxpy.utils.utils import phonopy_atoms_from_structure


def test_initialize_with_dataset(
    generate_workchain,
    generate_structure,
    generate_displacement_dataset,
    generate_settings,
    generate_force_sets,
    mock_calculator_code,
):
    """Test of PhonopyWorkChain.initialize() using NaCl data.

    `displacement_dataset` is generated using phonopy.

    """
    structure = generate_structure()
    settings = generate_settings()
    dataset = generate_displacement_dataset()
    force_sets = generate_force_sets().get_array("force_sets")
    inputs = {
        "structure": structure,
        "settings": settings,
        "calculator_inputs": {
            "force": {
                "code": mock_calculator_code("vasp.vasp"),
                "force_sets": force_sets,
            }
        },
    }

    wc = generate_workchain("phonoxpy.phonopy", inputs)
    wc.initialize()

    assert "displacement_dataset" not in wc.inputs
    phonon_setting_info_keys = [
        "version",
        "distance",
        "symmetry",
        "symmetry_tolerance",
        "primitive_matrix",
        "supercell_matrix",
    ]
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)
    _assert_dataset(wc, dataset)


def test_initialize_with_dataset_input(
    generate_workchain,
    generate_structure,
    generate_displacement_dataset,
    generate_settings,
    generate_force_sets,
    mock_calculator_code,
):
    """Test of PhonopyWorkChain.initialize() using NaCl data.

    `displacement_dataset` is given as an input.

    """
    structure = generate_structure()
    settings = generate_settings()
    dataset = generate_displacement_dataset()
    force_sets = generate_force_sets().get_array("force_sets")
    inputs = {
        "structure": structure,
        "settings": settings,
        "displacement_dataset": dataset,
        "calculator_inputs": {
            "force": {
                "code": mock_calculator_code("vasp.vasp"),
                "force_sets": force_sets,
            }
        },
    }
    wc = generate_workchain("phonoxpy.phonopy", inputs)
    wc.initialize()

    assert "displacement_dataset" in wc.inputs
    phonon_setting_info_keys = [
        "version",
        "symmetry",
        "symmetry_tolerance",
        "primitive_matrix",
        "supercell_matrix",
    ]
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)
    _assert_dataset(wc, dataset)


def _assert_dataset(wc, dataset):
    from aiida.orm import Dict, StructureData

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

    _assert_cells(wc)


def test_initialize_with_displacements(
    generate_workchain,
    generate_structure,
    generate_settings,
    generate_force_sets,
    mock_calculator_code,
):
    """Test of PhonopyWorkChain.initialize() using NaCl data.

    `displacements` (random displacements) is generated using phonopy.

    """
    structure = generate_structure()
    settings = generate_settings(number_of_snapshots=4)
    force_sets = generate_force_sets().get_array("force_sets")
    inputs = {
        "structure": structure,
        "settings": settings,
        "calculator_inputs": {
            "force": {
                "code": mock_calculator_code("vasp.vasp"),
                "force_sets": force_sets,
            }
        },
    }
    wc = generate_workchain("phonoxpy.phonopy", inputs)
    wc.initialize()
    assert "displacements" not in wc.inputs
    phonon_setting_info_keys = [
        "version",
        "distance",
        "symmetry",
        "symmetry_tolerance",
        "primitive_matrix",
        "supercell_matrix",
        "number_of_snapshots",
    ]
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)
    _assert_displacements(wc)


def test_initialize_with_displacements_input(
    generate_workchain,
    generate_structure,
    generate_displacements,
    generate_settings,
    generate_force_sets,
    mock_calculator_code,
):
    """Test of PhonopyWorkChain.initialize() using NaCl data.

    `displacements` (random displacements) is given as an input.

    """
    structure = generate_structure()
    settings = generate_settings()
    displacements = generate_displacements()
    force_sets = generate_force_sets().get_array("force_sets")
    inputs = {
        "structure": structure,
        "settings": settings,
        "displacements": displacements,
        "calculator_inputs": {
            "force": {
                "code": mock_calculator_code("vasp.vasp"),
                "force_sets": force_sets,
            }
        },
    }
    wc = generate_workchain("phonoxpy.phonopy", inputs)
    wc.initialize()
    assert "displacements" in wc.inputs
    np.testing.assert_almost_equal(
        wc.inputs.displacements.get_array("displacements"),
        wc.ctx.displacements.get_array("displacements"),
    )
    np.testing.assert_almost_equal(
        displacements.get_array("displacements"),
        wc.ctx.displacements.get_array("displacements"),
    )
    phonon_setting_info_keys = [
        "version",
        "symmetry",
        "symmetry_tolerance",
        "primitive_matrix",
        "supercell_matrix",
    ]
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)
    _assert_displacements(wc)


def _assert_displacements(wc):
    from aiida.orm import ArrayData, Dict, StructureData

    ctx = {
        "phonon_setting_info": Dict,
        "primitive": StructureData,
        "supercell": StructureData,
        "displacements": ArrayData,
        "supercells": dict,
    }
    for key in wc.ctx:
        assert key in ctx
        assert isinstance(wc.ctx[key], ctx[key])

    _assert_cells(wc)

    disps = wc.ctx.displacements.get_array("displacements")
    ndigits = len(str(len(disps)))
    for i, d in enumerate(disps):
        num = str(i + 1).zfill(ndigits)
        key = f"supercell_{num}"
        assert key in wc.ctx.supercells
        scell_per = phonopy_atoms_from_structure(wc.ctx.supercell)
        scell_per.positions = scell_per.positions + d
        scell_disp = phonopy_atoms_from_structure(wc.ctx.supercells[key])
        assert isclose(scell_disp, scell_per)


def _assert_cells(wc):
    primitive_matrix = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    np.testing.assert_almost_equal(
        wc.ctx.phonon_setting_info["primitive_matrix"], primitive_matrix
    )
    np.testing.assert_equal(
        wc.ctx.phonon_setting_info["supercell_matrix"], supercell_matrix
    )


def test_initialize_with_displacements_and_force_sets_input(
    generate_workchain,
    generate_structure,
    generate_displacements,
    generate_settings,
    generate_force_sets,
):
    """Test of PhonopyWorkChain.initialize() using NaCl data.

    `displacements` (random displacements) is given as an input.

    """
    from aiida.orm import ArrayData, Dict, StructureData

    structure = generate_structure()
    settings = generate_settings()
    displacements = generate_displacements()
    force_sets = generate_force_sets()
    inputs = {
        "structure": structure,
        "settings": settings,
        "displacements": displacements,
        "force_sets": force_sets,
    }
    wc = generate_workchain("phonoxpy.phonopy", inputs)
    wc.initialize()
    assert "displacements" in wc.inputs
    np.testing.assert_almost_equal(
        wc.inputs.displacements.get_array("displacements"),
        wc.ctx.displacements.get_array("displacements"),
    )
    np.testing.assert_almost_equal(
        displacements.get_array("displacements"),
        wc.ctx.displacements.get_array("displacements"),
    )
    phonon_setting_info_keys = [
        "version",
        "symmetry",
        "symmetry_tolerance",
        "primitive_matrix",
        "supercell_matrix",
    ]
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)
    ctx = {
        "phonon_setting_info": Dict,
        "primitive": StructureData,
        "supercell": StructureData,
        "displacements": ArrayData,
        "supercells": dict,
    }
    for key in wc.ctx:
        assert key in ctx
        assert isinstance(wc.ctx[key], ctx[key])
        assert "supercell_" not in key

    _assert_cells(wc)


def test_initialize_with_force_constants(
    generate_workchain, generate_structure, generate_settings, generate_fc_filedata
):
    """Test of PhonopyWorkChain.initialize() using NaCl data.

    `force_constants` is given as an input.

    """
    from aiida.orm import Dict, StructureData

    structure = generate_structure()
    settings = generate_settings()
    force_constants = generate_fc_filedata()
    inputs = {
        "structure": structure,
        "settings": settings,
        "force_constants": force_constants,
    }
    wc = generate_workchain("phonoxpy.phonopy", inputs)
    wc.initialize()
    assert "force_constants" in wc.inputs
    phonon_setting_info_keys = [
        "version",
        "symmetry",
        "symmetry_tolerance",
        "primitive_matrix",
        "supercell_matrix",
    ]
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)
    ctx = {
        "phonon_setting_info": Dict,
        "primitive": StructureData,
        "supercell": StructureData,
        "supercells": dict,
    }
    for key in wc.ctx:
        assert key in ctx
        assert isinstance(wc.ctx[key], ctx[key])
        assert "supercell_" not in key

    _assert_cells(wc)


def test_initialize_with_force_constants_for_random_disps(
    generate_workchain, generate_structure, generate_settings, generate_fc_filedata
):
    """Test of PhonopyWorkChain.initialize() using NaCl data.

    `force_constants` is given as an input along with `temperature` and
    `number_of_snapshots`.

    """
    from aiida.orm import Dict, StructureData, ArrayData

    structure = generate_structure()
    settings = generate_settings(temperature=300, number_of_snapshots=10)
    force_constants = generate_fc_filedata()
    inputs = {
        "structure": structure,
        "settings": settings,
        "force_constants": force_constants,
    }
    wc = generate_workchain("phonoxpy.phonopy", inputs)
    wc.initialize()
    assert "force_constants" in wc.inputs
    phonon_setting_info_keys = [
        "version",
        "symmetry",
        "symmetry_tolerance",
        "primitive_matrix",
        "supercell_matrix",
        "number_of_snapshots",
        "temperature",
    ]
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)
    ctx = {
        "phonon_setting_info": Dict,
        "primitive": StructureData,
        "supercell": StructureData,
        "supercells": dict,
        "displacements": ArrayData,
    }
    assert len(wc.ctx.supercells) == 10
    for key in wc.ctx:
        assert key in ctx
        assert isinstance(wc.ctx[key], ctx[key])
        assert "supercell_" not in key

    _assert_cells(wc)


def test_launch_process_with_dataset_inputs_and_run_phonopy(
    generate_inputs_phonopy_wc, generate_workchain, generate_settings
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool

    inputs = generate_inputs_phonopy_wc()
    inputs["run_phonopy"] = Bool(True)
    inputs["remote_phonopy"] = Bool(False)
    inputs["settings"] = generate_settings(mesh=100)
    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)
    output_keys = [
        "band_structure",
        "total_dos",
        "force_constants",
        "phonon_setting_info",
        "primitive",
        "supercell",
        "thermal_properties",
    ]
    assert set(list(result)) == set(output_keys)


def test_launch_process_with_dataset_inputs_and_run_phonopy_without_mesh(
    generate_inputs_phonopy_wc, generate_workchain
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool

    inputs = generate_inputs_phonopy_wc()
    inputs["run_phonopy"] = Bool(True)
    inputs["remote_phonopy"] = Bool(False)
    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)
    output_keys = [
        "force_constants",
        "phonon_setting_info",
        "primitive",
        "supercell",
    ]
    assert set(list(result)) == set(output_keys)
    with result["force_constants"].open(mode="rb") as source:
        with tempfile.TemporaryFile() as target:
            shutil.copyfileobj(source, target)
            target.seek(0)
            with h5py.File(target) as f:
                force_constants = f["force_constants"][:]
    np.testing.assert_allclose(
        force_constants[0, 0],
        np.eye(3) * 1.509115333333334,
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        force_constants[-1, -1],
        np.eye(3) * 2.206136666666667,
        atol=1e-8,
        rtol=0,
    )


def test_launch_process_with_dataset_inputs_and_run_phonopy_with_fc_calculator(
    generate_inputs_phonopy_wc, generate_workchain, generate_settings
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool

    inputs = generate_inputs_phonopy_wc()
    inputs["run_phonopy"] = Bool(True)
    inputs["remote_phonopy"] = Bool(False)
    inputs["settings"] = generate_settings(
        mesh=100, fc_calculator="alm", fc_calculator_options="cutoff = 5"
    )
    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)
    output_keys = [
        "band_structure",
        "total_dos",
        "force_constants",
        "phonon_setting_info",
        "primitive",
        "supercell",
        "thermal_properties",
    ]
    assert set(list(result)) == set(output_keys)


def test_launch_process_with_force_constants_and_run_phonopy(
    generate_inputs_phonopy_wc,
    generate_workchain,
    generate_settings,
    generate_fc_filedata,
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool

    inputs = generate_inputs_phonopy_wc()
    inputs["force_constants"] = generate_fc_filedata()
    inputs["run_phonopy"] = Bool(True)
    inputs["remote_phonopy"] = Bool(False)
    inputs["settings"] = generate_settings(mesh=100)
    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)
    output_keys = [
        "band_structure",
        "total_dos",
        "phonon_setting_info",
        "primitive",
        "supercell",
        "thermal_properties",
    ]
    assert set(list(result)) == set(output_keys)


def test_launch_process_with_dataset_inputs(
    generate_inputs_phonopy_wc, generate_workchain
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch

    inputs = generate_inputs_phonopy_wc()
    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)
    output_keys = ["phonon_setting_info", "primitive", "supercell"]
    assert set(list(result)) == set(output_keys)


def test_launch_process_with_displacements_inputs(
    fixture_code,
    generate_structure,
    generate_settings,
    generate_workchain,
    generate_force_sets,
    generate_displacements,
    generate_nac_params,
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool

    inputs = {
        "code": fixture_code("phonoxpy.phonopy"),
        "structure": generate_structure(),
        "settings": generate_settings(),
        "metadata": {},
        "force_sets": generate_force_sets("NaCl-displacements"),
        "displacements": generate_displacements(),
        "nac_params": generate_nac_params(),
        "run_phonopy": Bool(False),
    }
    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)
    output_keys = ["phonon_setting_info", "primitive", "supercell"]
    assert set(list(result)) == set(output_keys)


def test_launch_process_with_force_constants_inputs(
    fixture_code,
    generate_structure,
    generate_settings,
    generate_workchain,
    generate_nac_params,
    generate_fc_filedata,
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool

    inputs = {
        "code": fixture_code("phonoxpy.phonopy"),
        "structure": generate_structure(),
        "settings": generate_settings(),
        "metadata": {},
        "force_constants": generate_fc_filedata(),
        "nac_params": generate_nac_params(),
        "run_phonopy": Bool(False),
    }
    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)
    output_keys = ["phonon_setting_info", "primitive", "supercell"]
    assert set(list(result)) == set(output_keys)


@pytest.mark.parametrize("plugin_name", ["vasp.vasp", "quantumespresso.pw"])
def test_passing_through_ForcesWorkChain(
    fixture_code,
    generate_structure,
    generate_settings,
    generate_workchain,
    generate_force_sets,
    mock_calculator_code,
    plugin_name,
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool

    force_sets = generate_force_sets().get_array("force_sets")
    inputs = {
        "code": fixture_code("phonoxpy.phonopy"),
        "structure": generate_structure(),
        "settings": generate_settings(),
        "metadata": {},
        "calculator_inputs": {
            "force": {
                "code": mock_calculator_code(plugin_name),
                "force_sets": force_sets,
            }
        },
        "run_phonopy": Bool(False),
    }

    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)

    np.testing.assert_allclose(
        result["force_sets"].get_array("force_sets"),
        force_sets,
        atol=1e-8,
        rtol=0,
    )
    output_keys = (
        "phonon_setting_info",
        "primitive",
        "supercell",
        "displacement_dataset",
        "force_sets",
    )
    assert set(list(result)) == set(output_keys)


@pytest.mark.parametrize("plugin_name", ["vasp.vasp", "quantumespresso.pw"])
def test_passing_through_ForcesWorkChain_subtract_residual_forces(
    fixture_code,
    generate_structure,
    generate_settings,
    generate_workchain,
    generate_force_sets,
    mock_calculator_code,
    plugin_name,
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool

    force_sets = generate_force_sets().get_array("force_sets")
    supercell_force_set = np.zeros(force_sets.shape[1:], dtype=force_sets.dtype)

    inputs = {
        "code": fixture_code("phonoxpy.phonopy"),
        "structure": generate_structure(),
        "settings": generate_settings(),
        "metadata": {},
        "calculator_inputs": {
            "force": {
                "code": mock_calculator_code(plugin_name),
                "force_sets": force_sets,
                "supercell_force_set": supercell_force_set,
            }
        },
        "run_phonopy": Bool(False),
        "subtract_residual_forces": Bool(True),
    }

    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)

    np.testing.assert_allclose(
        result["force_sets"].get_array("force_sets"),
        force_sets,
        atol=1e-8,
        rtol=0,
    )
    output_keys = (
        "phonon_setting_info",
        "primitive",
        "supercell",
        "displacement_dataset",
        "force_sets",
        "supercell_forces",
    )
    assert set(list(result)) == set(output_keys)


@pytest.mark.parametrize("plugin_name", ["vasp.vasp", "quantumespresso.pw"])
def test_passing_through_NacParamsWorkChain(
    fixture_code,
    generate_structure,
    generate_settings,
    generate_workchain,
    generate_force_sets,
    generate_nac_params,
    mock_calculator_code,
    plugin_name,
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool

    nac_params_data = generate_nac_params()
    born_charges = nac_params_data.get_array("born_charges")
    epsilon = nac_params_data.get_array("epsilon")
    force_sets = generate_force_sets().get_array("force_sets")

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
        "code": fixture_code("phonoxpy.phonopy"),
        "structure": generate_structure(),
        "settings": generate_settings(),
        "metadata": {},
        "calculator_inputs": {
            "force": {
                "code": mock_calculator_code(plugin_name),
                "force_sets": force_sets,
            },
            "nac": nac_inputs,
        },
        "run_phonopy": Bool(False),
    }

    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)

    for key in ("born_charges", "epsilon"):
        np.testing.assert_allclose(
            result["nac_params"].get_array(key),
            generate_nac_params().get_array(key),
            atol=1e-8,
            rtol=0,
        )

    output_keys = (
        "phonon_setting_info",
        "primitive",
        "supercell",
        "displacement_dataset",
        "force_sets",
        "nac_params",
    )
    assert set(list(result)) == set(output_keys)


@pytest.mark.parametrize("plugin_name", ["vasp.vasp", "quantumespresso.pw"])
def test_passing_through_NacParamsWorkChain_without_force(
    fixture_code,
    generate_structure,
    generate_settings,
    generate_workchain,
    generate_nac_params,
    mock_calculator_code,
    plugin_name,
):
    """Test of PhonopyWorkChain using NaCl data.

    NAC only calculation without `inputs.calculator_inputs["force"]`.
    When `supercell_matrix` is given in `inputs.settings`, supercells and displacements
    are generated.

    """
    from aiida.engine import launch
    from aiida.orm import Bool

    nac_params_data = generate_nac_params()
    born_charges = nac_params_data.get_array("born_charges")
    epsilon = nac_params_data.get_array("epsilon")

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
        "code": fixture_code("phonoxpy.phonopy"),
        "structure": generate_structure(),
        "settings": generate_settings(),
        "metadata": {},
        "calculator_inputs": {"nac": nac_inputs},
        "run_phonopy": Bool(False),
    }

    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)

    for key in ("born_charges", "epsilon"):
        np.testing.assert_allclose(
            result["nac_params"].get_array(key),
            generate_nac_params().get_array(key),
            atol=1e-8,
            rtol=0,
        )

    output_keys = (
        "phonon_setting_info",
        "primitive",
        "supercell",
        "displacement_dataset",
        "nac_params",
    )
    assert set(list(result)) == set(output_keys)


@pytest.mark.parametrize("plugin_name", ["vasp.vasp", "quantumespresso.pw"])
def test_passing_through_NacParamsWorkChain_with_no_settings_input(
    fixture_code,
    generate_structure,
    generate_workchain,
    generate_nac_params,
    mock_calculator_code,
    plugin_name,
):
    """Test of PhonopyWorkChain using NaCl data.

    NAC only calculation without `inputs.settings`.
    When `supercell_matrix` is not in `inputs.settings`, supercells and displacements
    are not generated. When `inputs.settings`, the default value of `Dict(dict={})` is
    given to `PhonopyWorkChain`.

    """
    from aiida.engine import launch
    from aiida.orm import Bool

    nac_params_data = generate_nac_params()
    born_charges = nac_params_data.get_array("born_charges")
    epsilon = nac_params_data.get_array("epsilon")

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
        "code": fixture_code("phonoxpy.phonopy"),
        "structure": generate_structure(),
        "metadata": {},
        "calculator_inputs": {"nac": nac_inputs},
        "run_phonopy": Bool(False),
    }

    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)

    for key in ("born_charges", "epsilon"):
        np.testing.assert_allclose(
            result["nac_params"].get_array(key),
            generate_nac_params().get_array(key),
            atol=1e-8,
            rtol=0,
        )

    output_keys = (
        "phonon_setting_info",
        "primitive",
        "nac_params",
    )
    assert set(list(result)) == set(output_keys)


@pytest.mark.parametrize("plugin_name", ["vasp.vasp"])
def test_passing_through_NacParamsWorkChain_nac_structure(
    fixture_code,
    generate_structure,
    generate_settings,
    generate_workchain,
    generate_force_sets,
    generate_nac_params,
    mock_calculator_code,
    plugin_name,
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool

    nac_params_data = generate_nac_params(structure_id="NaCl-unitcell")
    born_charges = nac_params_data.get_array("born_charges")
    epsilon = nac_params_data.get_array("epsilon")
    force_sets = generate_force_sets().get_array("force_sets")

    if plugin_name == "vasp.vasp":
        nac_inputs = {
            "code": mock_calculator_code(plugin_name),
            "born_charges": born_charges,
            "epsilon": epsilon,
        }
    else:
        raise RuntimeError("plugin_name doesn't exist.")

    inputs = {
        "code": fixture_code("phonoxpy.phonopy"),
        "structure": generate_structure(),
        "nac_structure": generate_structure(),
        "settings": generate_settings(),
        "metadata": {},
        "calculator_inputs": {
            "force": {
                "code": mock_calculator_code(plugin_name),
                "force_sets": force_sets,
            },
            "nac": nac_inputs,
        },
        "run_phonopy": Bool(False),
    }

    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)

    for key in ("born_charges", "epsilon"):
        np.testing.assert_allclose(
            result["nac_params"].get_array(key),
            generate_nac_params().get_array(key),
            atol=1e-8,
            rtol=0,
        )

    output_keys = (
        "phonon_setting_info",
        "primitive",
        "supercell",
        "displacement_dataset",
        "force_sets",
        "nac_params",
    )
    assert set(list(result)) == set(output_keys)
