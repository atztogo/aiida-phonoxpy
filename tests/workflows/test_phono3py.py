"""Tests for Phono3pyWorkChain."""
import numpy as np
import pytest

# from phonopy.structure.cells import isclose
# from aiida_phonoxpy.utils.utils import phonopy_atoms_from_structure


@pytest.mark.parametrize("plugin_name", ["vasp.vasp", "quantumespresso.pw"])
def test_passing_through_ForcesWorkChain(
    generate_structure,
    generate_phonopy_settings,
    generate_workchain,
    generate_force_sets,
    mock_calculator_code,
    mock_forces_run_calculation,
    plugin_name,
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch

    force_sets = generate_force_sets(structure_id="NaCl-fc3").get_array("force_sets")
    inputs = {
        "structure": generate_structure(),
        "settings": generate_phonopy_settings(),
        "metadata": {},
        "calculator_inputs": {
            "force": {
                "code": mock_calculator_code(plugin_name),
                "force_sets": force_sets,
            }
        },
    }

    mock_forces_run_calculation()
    process = generate_workchain("phonoxpy.phono3py", inputs)
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
