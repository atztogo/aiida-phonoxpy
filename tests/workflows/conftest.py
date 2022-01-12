"""Pytest fixtures for workflows."""
import numpy as np
import pytest


@pytest.fixture
def generate_workchain(
    mock_forces_run_calculation,
    mock_nac_params_run_calculation,
    mock_run_phono3py_fc,
    mock_run_phono3py_ltc,
):
    """Generate an instance of a `WorkChain`.

    Mocking workchain has to be done by fixtures as parameters of this method.

    """

    def _generate_workchain(entry_point, inputs):
        """Generate an instance of a `WorkChain` with the given entry point and inputs.

        Parameters
        ----------
        entry_point : str
            Entry point name of the work chain subclass.
        inputs : dict
            Inputs to be passed to process construction.

        Returns
        -------
        WorkChain

        """
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import WorkflowFactory

        process_class = WorkflowFactory(entry_point)
        runner = get_manager().get_runner()
        process = instantiate_process(runner, process_class, **inputs)

        return process

    return _generate_workchain


@pytest.fixture
def generate_inputs_phonopy_wc(
    fixture_code,
    generate_structure,
    generate_displacement_dataset,
    generate_force_sets,
    generate_nac_params,
    generate_settings,
):
    """Return inputs for phonopy workchain."""

    def _generate_inputs_phonopy(metadata=None):
        from aiida.orm import Bool

        return {
            "code": fixture_code("phonoxpy.phonopy"),
            "structure": generate_structure(),
            "settings": generate_settings(),
            "metadata": metadata or {},
            "force_sets": generate_force_sets(),
            "displacement_dataset": generate_displacement_dataset(),
            "nac_params": generate_nac_params(),
            "run_phonopy": Bool(False),
        }

    return _generate_inputs_phonopy


@pytest.fixture
def mock_forces_run_calculation(monkeypatch):
    """Return mock ForcesWorkChain.run_calculation method.

    VASP and QE-pw calculation outputs are mocked.

    `self.ctx.plugin_name` is determined by `inputs.calculator_inputs['code']`
    in ForceWorkChain. See ForceWorkChain.initialize()

    As the rule of this test, `force_sets` has to be passed through
    `inputs.calculator_inputs['force_sets']` in ForceWorkChain.

    """
    from aiida.plugins import WorkflowFactory

    ForcesWorkChain = WorkflowFactory("phonoxpy.forces")

    def _mock(self):
        """Mock method to replace ForceWorkChain.run_calculation.

        self.inputs.structure is a supercell with the label, e.g.,

        supercell_01, ...
        phonon_supercell_1, ...

        """
        from aiida.common import AttributeDict
        from aiida.orm import ArrayData

        label = self.inputs.structure.label
        forces_index = int(label.split("_")[-1]) - 1
        forces = ArrayData()
        self.ctx.calc = AttributeDict()
        self.ctx.calc.outputs = AttributeDict()
        force_sets = self.inputs.calculator_inputs["force_sets"]
        if self.ctx.plugin_name == "vasp.vasp":
            forces.set_array("final", np.array(force_sets[forces_index]))
            self.ctx.calc.outputs.forces = forces
        elif self.ctx.plugin_name == "quantumespresso.pw":
            forces.set_array("forces", np.array([force_sets[forces_index]]))
            self.ctx.calc.outputs.output_trajectory = forces

    monkeypatch.setattr(ForcesWorkChain, "run_calculation", _mock)


@pytest.fixture
def mock_nac_params_run_calculation(monkeypatch):
    """Return mock NacParamsWorkChain.run_calculation method.

    VASP and QE-{pw,ph} calculation outputs are mocked.

    `self.ctx.plugin_names` should be set via `inputs.calculator_inputs['code']`
    in NacParamsWorkChain. See NacParamsWorkChain.initialize()

    As the rule of this test, `born_charges` and `epsilon` have to be passed through
    `inputs.calculator_inputs` in NacParamsWorkChain.

    """
    from aiida.plugins import WorkflowFactory

    NacParamsWorkChain = WorkflowFactory("phonoxpy.nac_params")

    def _mock(self):
        from aiida.common import AttributeDict
        from aiida.orm import ArrayData, Dict

        born_charges = self.inputs.calculator_inputs["born_charges"]
        epsilon = self.inputs.calculator_inputs["epsilon"]

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


@pytest.fixture
def mock_run_phono3py_fc(monkeypatch, generate_fc3_filedata, generate_fc2_filedata):
    """Return mock Phono3pyFCWorkChain.run_phono3py_fc method."""
    from aiida.plugins import WorkflowFactory

    Phono3pyFCWorkChain = WorkflowFactory("phonoxpy.phono3py_fc")

    def _mock(self):
        """Mock method to replace Phono3pyFCWorkChain.run_phono3py_fc."""
        from aiida.common import AttributeDict

        self.ctx.fc_calc = AttributeDict()
        self.ctx.fc_calc.outputs = AttributeDict()
        self.ctx.fc_calc.outputs.fc3 = generate_fc3_filedata()
        self.ctx.fc_calc.outputs.fc2 = generate_fc2_filedata()

    monkeypatch.setattr(Phono3pyFCWorkChain, "run_phono3py_fc", _mock)


@pytest.fixture
def mock_run_phono3py_ltc(monkeypatch, generate_ltc_filedata):
    """Return mock Phono3pyLTCWorkChain.run_phono3py_ltc method."""
    from aiida.plugins import WorkflowFactory

    Phono3pyLTCWorkChain = WorkflowFactory("phonoxpy.phono3py_ltc")

    def _mock(self):
        """Mock method to replace Phono3pyLTCWorkChain.run_phono3py_ltc."""
        from aiida.common import AttributeDict

        self.ctx.ltc_calc = AttributeDict()
        self.ctx.ltc_calc.outputs = AttributeDict()
        self.ctx.ltc_calc.outputs.ltc = generate_ltc_filedata()

    monkeypatch.setattr(Phono3pyLTCWorkChain, "run_phono3py_ltc", _mock)
