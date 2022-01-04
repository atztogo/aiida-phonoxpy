"""Tests for PhonopyWorkChain."""
import numpy as np
import pytest
from phonopy.structure.cells import isclose

from aiida_phonoxpy.utils.utils import phonopy_atoms_from_structure


@pytest.fixture
def mock_forces_run_calculation(generate_force_sets, monkeypatch):
    """Return mock ForcesWorkChain.run_calculation method.

    VASP and QE-pw calculation outputs are mocked.

    `self.ctx.plugin_name` should be set via `inputs.code`. See
    ForceWorkChain.initialize()

    """

    def _mock_forces_run_calculation(structure_id="NaCl"):
        from aiida.plugins import WorkflowFactory

        ForcesWorkChain = WorkflowFactory("phonoxpy.forces")
        force_sets_data = generate_force_sets(structure_id=structure_id)
        force_sets = force_sets_data.get_array("force_sets")

        def _mock(self):
            from aiida.orm import ArrayData
            from aiida.common import AttributeDict

            label = self.inputs.structure.label
            forces_index = int(label.split("_")[1]) - 1
            forces = ArrayData()
            self.ctx.calc = AttributeDict()
            self.ctx.calc.outputs = AttributeDict()

            if self.ctx.plugin_name == "vasp.vasp":
                forces.set_array("final", np.array(force_sets[forces_index]))
                self.ctx.calc.outputs.forces = forces
            elif self.ctx.plugin_name == "quantumespresso.pw":
                forces.set_array("forces", np.array(force_sets[forces_index]))
                self.ctx.calc.outputs.output_trajectory = forces

        monkeypatch.setattr(ForcesWorkChain, "run_calculation", _mock)

    return _mock_forces_run_calculation


@pytest.fixture
def mock_nac_params_run_calculation(generate_nac_params, monkeypatch):
    """Return mock NacParamsWorkChain.run_calculation method.

    VASP and QE-{pw,ph} calculation outputs are mocked.

    `self.ctx.plugin_names[0]` should be set via `inputs.code`. See
    NacParamsWorkChain.initialize()

    """

    def _mock_nac_params_run_calculation(structure_id="NaCl"):
        from aiida.plugins import WorkflowFactory

        NacParamsWorkChain = WorkflowFactory("phonoxpy.nac_params")
        nac_params_data = generate_nac_params(structure_id=structure_id)
        born_charges = nac_params_data.get_array("born_charges")
        epsilon = nac_params_data.get_array("epsilon")

        def _mock(self):
            from aiida.orm import ArrayData, Dict
            from aiida.common import AttributeDict

            if self.ctx.plugin_names[0] == "vasp.vasp":
                calc = AttributeDict()
                calc.inputs = AttributeDict()
                calc.outputs = AttributeDict()
                calc.inputs.structure = self.inputs.structure
                born_charges_data = ArrayData()
                born_charges_data.set_array("born_charges", born_charges)
                epsilon_data = ArrayData()
                epsilon_data.set_array("epsilon", epsilon)
                calc.outputs.born_charges = born_charges_data
                calc.outputs.dielectrics = epsilon_data
                self.ctx.nac_params_calcs = [calc]
            elif self.ctx.plugin_names[0] == "quantumespresso.pw":
                if self.ctx.iteration == 1:
                    assert self.ctx.plugin_names[0] == "quantumespresso.pw"
                    pw_calc = AttributeDict()
                    pw_calc.inputs = AttributeDict()
                    pw_calc.inputs.pw = AttributeDict()
                    pw_calc.inputs.pw.structure = self.inputs.structure
                    self.ctx.nac_params_calcs = [pw_calc]
                elif self.ctx.iteration == 2:
                    assert self.ctx.plugin_names[1] == "quantumespresso.ph"
                    ph_calc = AttributeDict()
                    ph_calc.outputs = AttributeDict()
                    ph_calc.outputs.output_parameters = Dict(
                        dict={
                            "effective_charges_eu": born_charges,
                            "dielectric_constant": epsilon,
                        }
                    )
                    self.ctx.nac_params_calcs.append(ph_calc)

        monkeypatch.setattr(NacParamsWorkChain, "run_calculation", _mock)

    return _mock_nac_params_run_calculation


def test_initialize_with_dataset(
    generate_workchain,
    generate_structure,
    generate_displacement_dataset,
    generate_phonopy_settings,
):
    """Test of PhonopyWorkChain.initialize() using NaCl data.

    `displacement_dataset` is generated using phonopy.

    """
    structure = generate_structure()
    settings = generate_phonopy_settings()
    dataset = generate_displacement_dataset()
    inputs = {"structure": structure, "settings": settings}
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
    generate_phonopy_settings,
):
    """Test of PhonopyWorkChain.initialize() using NaCl data.

    `displacement_dataset` is given as an input.

    """
    structure = generate_structure()
    settings = generate_phonopy_settings()
    dataset = generate_displacement_dataset()
    inputs = {
        "structure": structure,
        "settings": settings,
        "displacement_dataset": dataset,
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
    generate_phonopy_settings,
):
    """Test of PhonopyWorkChain.initialize() using NaCl data.

    `displacements` (random displacements) is generated using phonopy.

    """
    structure = generate_structure()
    settings = generate_phonopy_settings(number_of_snapshots=4)
    inputs = {"structure": structure, "settings": settings}
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
    generate_phonopy_settings,
):
    """Test of PhonopyWorkChain.initialize() using NaCl data.

    `displacements` (random displacements) is given as an input.

    """
    structure = generate_structure()
    settings = generate_phonopy_settings()
    displacements = generate_displacements()
    inputs = {
        "structure": structure,
        "settings": settings,
        "displacements": displacements,
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
    for i, d in enumerate(disps):
        ndigits = len(str(len(disps)))
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


def test_launch_process_with_dataset_inputs_and_run_phonopy(
    generate_inputs_phonopy_wc, generate_workchain
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool

    inputs = generate_inputs_phonopy_wc()
    inputs["run_phonopy"] = Bool(True)
    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)
    output_keys = [
        "band_structure",
        "dos",
        "force_constants",
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
    generate_phonopy_settings,
    generate_workchain,
    generate_force_sets,
    generate_displacements,
    generate_nac_params,
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch

    inputs = {
        "code": fixture_code("phonoxpy.phonopy"),
        "structure": generate_structure(),
        "settings": generate_phonopy_settings(),
        "metadata": {},
        "force_sets": generate_force_sets("NaCl-displacements"),
        "displacements": generate_displacements(),
        "nac_params": generate_nac_params(),
    }
    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)
    output_keys = ["phonon_setting_info", "primitive", "supercell"]
    assert set(list(result)) == set(output_keys)


@pytest.mark.parametrize("plugin_name", ["vasp.vasp", "quantumespresso.pw"])
def test_passing_through_ForcesWorkChain(
    fixture_code,
    generate_structure,
    generate_phonopy_settings,
    generate_workchain,
    mock_calculator_code,
    mock_forces_run_calculation,
    plugin_name,
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool

    mock_forces_run_calculation()

    inputs = {
        "code": fixture_code("phonoxpy.phonopy"),
        "structure": generate_structure(),
        "settings": generate_phonopy_settings(),
        "metadata": {},
        "calculator_inputs": {"force": {"code": mock_calculator_code(plugin_name)}},
        "run_phonopy": Bool(False),
    }
    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)


@pytest.mark.parametrize("plugin_name", ["vasp.vasp", "quantumespresso.pw"])
def test_passing_through_NacParamsWorkChain(
    fixture_code,
    generate_structure,
    generate_phonopy_settings,
    generate_workchain,
    mock_calculator_code,
    mock_forces_run_calculation,
    mock_nac_params_run_calculation,
    plugin_name,
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Bool

    mock_forces_run_calculation()
    mock_nac_params_run_calculation()

    if plugin_name == "vasp.vasp":
        nac_inputs = {"code": mock_calculator_code(plugin_name)}
    elif plugin_name == "quantumespresso.pw":
        nac_inputs = {
            "steps": [
                {"code": mock_calculator_code("quantumespresso.pw")},
                {"code": mock_calculator_code("quantumespresso.ph")},
            ]
        }
    else:
        raise RuntimeError("plugin_name doesn't exist.")

    inputs = {
        "code": fixture_code("phonoxpy.phonopy"),
        "structure": generate_structure(),
        "settings": generate_phonopy_settings(),
        "metadata": {},
        "calculator_inputs": {
            "force": {"code": mock_calculator_code(plugin_name)},
            "nac": nac_inputs,
        },
        "run_phonopy": Bool(False),
    }
    process = generate_workchain("phonoxpy.phonopy", inputs)
    result, node = launch.run_get_node(process)
