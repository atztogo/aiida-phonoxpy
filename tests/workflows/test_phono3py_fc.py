"""Tests for Phono3pyFCWorkChain."""


def test_initialize(
    generate_workchain,
    generate_structure,
    generate_settings,
):
    """Test of Phono3pyFCWorkChain.initialize() using NaCl data."""
    structure = generate_structure()
    settings = generate_settings()

    inputs = {
        "structure": structure,
        "settings": settings,
    }
    wc = generate_workchain("phonoxpy.phono3py_fc", inputs)
    wc.initialize()

    phonon_setting_info_keys = [
        "version",
        "supercell_matrix",
        "symmetry_tolerance",
    ]
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)


def test_initialize_with_phonon_supercell_matrix(
    generate_workchain,
    generate_structure,
    generate_settings,
):
    """Test of Phono3pyFCWorkChain.initialize() using NaCl data.

    With phonon_supercell_matrix.

    """
    structure = generate_structure()
    settings = generate_settings(phonon_supercell_matrix=[2, 2, 2])

    inputs = {
        "structure": structure,
        "settings": settings,
    }
    wc = generate_workchain("phonoxpy.phono3py_fc", inputs)
    wc.initialize()

    phonon_setting_info_keys = [
        "version",
        "supercell_matrix",
        "phonon_supercell_matrix",
        "symmetry_tolerance",
    ]
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)


def test_Phono3pyFCWorkChain_full(
    generate_structure,
    generate_workchain,
    generate_displacement_dataset,
    generate_force_sets,
):
    """Test of Phono3pyFCWorkChain using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Dict

    # mock_run_phono3py_fc()

    settings = {
        "supercell_matrix": [1, 1, 1],
        "phonon_supercell_matrix": [2, 2, 2],
    }

    inputs = {
        "structure": generate_structure(),
        "settings": Dict(dict=settings),
        "metadata": {},
        "displacement_dataset": generate_displacement_dataset(structure_id="NaCl-fc3"),
        "force_sets": generate_force_sets(structure_id="NaCl-fc3"),
        "phonon_displacement_dataset": generate_displacement_dataset(
            structure_id="NaCl-64"
        ),
        "phonon_force_sets": generate_force_sets(structure_id="NaCl-64"),
    }

    process = generate_workchain("phonoxpy.phono3py_fc", inputs)
    results, node = launch.run_get_node(process)

    output_keys = ("fc2", "fc3", "phonon_setting_info")
    assert set(list(results)) == set(output_keys)


def test_Phono3pyFCWorkChain_full_with_fc_calculator(
    generate_structure,
    generate_workchain,
    generate_displacement_dataset,
    generate_force_sets,
):
    """Test of Phono3pyFCWorkChain using NaCl data with ALM."""
    from aiida.engine import launch
    from aiida.orm import Dict

    # mock_run_phono3py_fc()

    settings = {
        "supercell_matrix": [1, 1, 1],
        "phonon_supercell_matrix": [2, 2, 2],
        "fc_calculator": "alm",
        "fc_calculator_options": "cutoff = 5",
    }

    inputs = {
        "structure": generate_structure(),
        "settings": Dict(dict=settings),
        "metadata": {},
        "displacement_dataset": generate_displacement_dataset(structure_id="NaCl-fc3"),
        "force_sets": generate_force_sets(structure_id="NaCl-fc3"),
        "phonon_displacement_dataset": generate_displacement_dataset(
            structure_id="NaCl-64"
        ),
        "phonon_force_sets": generate_force_sets(structure_id="NaCl-64"),
    }

    process = generate_workchain("phonoxpy.phono3py_fc", inputs)
    results, node = launch.run_get_node(process)

    output_keys = ("fc2", "fc3", "phonon_setting_info")
    assert set(list(results)) == set(output_keys)
    assert "fc_calculator" in results["phonon_setting_info"]
    assert results["phonon_setting_info"]["fc_calculator"] == "alm"
    assert "fc_calculator_options" in results["phonon_setting_info"]
    assert results["phonon_setting_info"]["fc_calculator_options"] == "cutoff = 5"
