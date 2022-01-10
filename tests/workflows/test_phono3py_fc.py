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


def test_Phono3pyFCWorkChain_full(
    generate_structure,
    generate_workchain,
    generate_displacement_dataset,
    generate_force_sets,
    generate_fc3_filedata,
    generate_fc2_filedata,
    monkeypatch,
):
    """Test of PhonopyWorkChain with dataset inputs using NaCl data."""
    from aiida.engine import launch
    from aiida.orm import Dict
    from aiida.plugins import WorkflowFactory

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

    Phono3pyFCWorkChain = WorkflowFactory("phonoxpy.phono3py_fc")

    def _mock(self):
        """Mock method to replace Phono3pyFCWorkChain.run_phono3py_fc."""
        from aiida.common import AttributeDict

        self.ctx.fc_calc = AttributeDict()
        self.ctx.fc_calc.outputs = AttributeDict()
        self.ctx.fc_calc.outputs.fc3 = generate_fc3_filedata()
        self.ctx.fc_calc.outputs.fc2 = generate_fc2_filedata()

    monkeypatch.setattr(Phono3pyFCWorkChain, "run_phono3py_fc", _mock)

    process = generate_workchain("phonoxpy.phono3py_fc", inputs)
    results, node = launch.run_get_node(process)

    output_keys = ("fc2", "fc3", "phonon_setting_info")
    assert set(list(results)) == set(output_keys)
