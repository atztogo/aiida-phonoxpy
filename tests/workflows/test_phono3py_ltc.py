"""Tests for Phono3pyLTCWorkChain."""


def test_initialize(
    generate_workchain,
    generate_structure,
    generate_settings,
):
    """Test of Phono3pyLTCWorkChain.initialize() using NaCl data."""
    structure = generate_structure()
    settings = generate_settings()

    inputs = {
        "structure": structure,
        "settings": settings,
    }
    wc = generate_workchain("phonoxpy.phono3py_ltc", inputs)
    wc.initialize()

    phonon_setting_info_keys = [
        "version",
        "supercell_matrix",
        "symmetry_tolerance",
        "symmetry",
        "primitive_matrix",
    ]
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)


def test_initialize_with_isotope(
    generate_workchain,
    generate_structure,
    generate_settings,
):
    """Test of Phono3pyLTCWorkChain.initialize() using NaCl data."""
    structure = generate_structure()
    settings = generate_settings(isotope=True)

    inputs = {
        "structure": structure,
        "settings": settings,
    }
    wc = generate_workchain("phonoxpy.phono3py_ltc", inputs)
    wc.initialize()

    phonon_setting_info_keys = [
        "version",
        "supercell_matrix",
        "symmetry_tolerance",
        "symmetry",
        "primitive_matrix",
        "isotope",
    ]
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)


def test_initialize_with_phonon_supercell_matrix(
    generate_workchain,
    generate_structure,
    generate_settings,
):
    """Test of Phono3pyLTCWorkChain.initialize() using NaCl data.

    With phonon_supercell_matrix.

    """
    structure = generate_structure()
    settings = generate_settings(phonon_supercell_matrix=[2, 2, 2])

    inputs = {
        "structure": structure,
        "settings": settings,
    }
    wc = generate_workchain("phonoxpy.phono3py_ltc", inputs)
    wc.initialize()

    phonon_setting_info_keys = [
        "version",
        "supercell_matrix",
        "phonon_supercell_matrix",
        "symmetry_tolerance",
        "symmetry",
        "primitive_matrix",
    ]
    assert set(wc.ctx.phonon_setting_info.keys()) == set(phonon_setting_info_keys)


def test_Phono3pyLTCWorkChain_full(
    generate_structure,
    generate_workchain,
    generate_fc3_filedata,
    generate_fc2_filedata,
    generate_nac_params,
):
    """Test of Phono3pyLTCWorkChain using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Dict

    # mock_run_phono3py_fc()

    settings = {
        "supercell_matrix": [2, 2, 2],
        "phonon_supercell_matrix": [4, 4, 4],
    }

    inputs = {
        "structure": generate_structure(),
        "settings": Dict(dict=settings),
        "metadata": {},
        "fc2": generate_fc2_filedata(structure_id="NaCl-512"),
        "fc3": generate_fc3_filedata(structure_id="NaCl-64"),
        "nac_params": generate_nac_params(),
    }

    process = generate_workchain("phonoxpy.phono3py_ltc", inputs)
    results, node = launch.run_get_node(process)
    output_keys = ("ltc", "phonon_setting_info")
    assert set(list(results)) == set(output_keys)
