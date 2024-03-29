"""Workflow to calculate supercell forces."""

import numpy as np
from aiida.engine import WorkChain, calcfunction, if_
from aiida.orm import ArrayData, Float, StructureData
from aiida.plugins import WorkflowFactory

from aiida_phonoxpy.common.builders import (
    get_calculator_process,
    get_import_workchain_inputs,
    get_plugin_names,
    get_workchain_inputs,
)
from aiida_phonoxpy.utils.utils import (
    compare_structures,
    get_structure_from_vasp_immigrant,
)
from aiida_phonoxpy.workflows.mixin import DoNothingMixIn


def _get_forces(outputs, plugin_name):
    """Return supercell force ArrayData."""
    if plugin_name == "vasp.vasp":
        if "forces" in outputs and "final" in outputs.forces.get_arraynames():
            forces_data = get_vasp_forces(outputs.forces)
        else:
            return None
    elif plugin_name == "quantumespresso.pw":
        if (
            "output_trajectory" in outputs
            and "forces" in outputs.output_trajectory.get_arraynames()
        ):
            forces_data = get_qe_forces(outputs.output_trajectory)
        else:
            return None
    return forces_data


@calcfunction
def get_vasp_forces(forces):
    """Return VASP forces ArrayData."""
    forces_data = ArrayData()
    forces_data.set_array("forces", forces.get_array("final"))
    forces_data.label = "forces"
    return forces_data


@calcfunction
def get_qe_forces(output_trajectory):
    """Return QE forces ArrayData."""
    forces_data = ArrayData()
    forces_data.set_array("forces", output_trajectory.get_array("forces")[-1])
    forces_data.label = "forces"
    return forces_data


def _get_energy(outputs, plugin_name):
    """Return supercell energy ArrayData."""
    if plugin_name == "vasp.vasp":
        ekey = "energy_extrapolated"
        if "energies" in outputs and ekey in outputs.energies.get_arraynames():
            return get_vasp_energy(outputs.energies)
        else:
            return None

    if plugin_name == "quantumespresso.pw":
        if (
            "output_parameters" in outputs
            and "energy" in outputs.output_parameters.keys()
        ):
            return get_qe_energy(outputs.output_parameters)
        else:
            return None


@calcfunction
def get_vasp_energy(energies):
    """Return VASP energy ArrayData.

    energies is an 1D-array of energies.

    {'electronic_step_energies': True} gives SC energies.
    {'electronic_step_energies': False} gives last SC energy.

    """
    energy_data = ArrayData()
    ekey = "energy_extrapolated"
    energies = energies.get_array(ekey)
    energy_data.set_array("energy", np.array(energies, dtype=float))
    energy_data.label = "energy"
    return energy_data


@calcfunction
def get_qe_energy(output_parameters):
    """Return VASP energy ArrayData."""
    energy_data = ArrayData()
    energy_data.set_array(
        "energy",
        np.array(
            [
                output_parameters["energy"],
            ],
            dtype=float,
        ),
    )
    energy_data.label = "energy"
    return energy_data


class ForcesWorkChain(WorkChain, DoNothingMixIn):
    """Wrapper to compute supercell forces."""

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline.

        remote_workdir : str
            This is used to import VASP calculation from VASP output files. The data
            are imported from remote_workdir of computer.

        """
        super().define(spec)
        spec.input("structure", valid_type=StructureData, required=True)
        spec.input("calculator_inputs", valid_type=dict, required=True, non_db=True)
        spec.input("symmetry_tolerance", valid_type=Float, default=lambda: Float(1e-5))
        spec.input("donothing_inputs", valid_type=dict, required=False, non_db=True)
        spec.outline(
            cls.initialize,
            if_(cls.import_calculation)(
                cls.import_calculation_from_workdir,
                cls.validate_imported_structure,
            ).else_(
                if_(cls.use_donothing)(
                    cls.do_nothing,
                ),
                cls.run_calculation,
            ),
            cls.finalize,
        )

        spec.output("forces", valid_type=ArrayData, required=True)
        spec.output("energy", valid_type=ArrayData, required=False)

        spec.exit_code(
            1001,
            "ERROR_NO_FORCES",
            message="forces could not be retrieved from calculaton.",
        )
        spec.exit_code(
            1002,
            "ERROR_NO_ENERGY",
            message="energy could not be retrieved from calculaton.",
        )
        spec.exit_code(
            1003,
            "ERROR_STRUCTURE_VALIDATION",
            message="input and imported structures are different.",
        )
        spec.exit_code(
            1004,
            "ERROR_WAITING_TIMEOUT",
            message="time out in queue waiting.",
        )

    def import_calculation(self):
        """Return boolean for outline."""
        return "remote_workdir" in self.inputs.calculator_inputs

    def initialize(self):
        """Initialize outline control parameters."""
        self.report("initialization")
        self.ctx.plugin_name = get_plugin_names(self.inputs.calculator_inputs)[0]

    def run_calculation(self):
        """Run supercell force calculation."""
        self.report("calculate supercell forces")
        process_inputs = get_workchain_inputs(
            self.inputs.calculator_inputs,
            self.inputs.structure,
            label=self.metadata.label,
        )
        CalculatorProcess = get_calculator_process(self.ctx.plugin_name)
        future = self.submit(CalculatorProcess, **process_inputs)
        self.report("{} pk = {}".format(self.metadata.label, future.pk))
        self.to_context(**{"calc": future})

    def import_calculation_from_workdir(self):
        """Import supercell force calculation.

        Only VaspImmigrantWorkChain is supported.

        """
        self.report("import supercell force calculation data in files.")
        inputs = get_import_workchain_inputs(
            self.inputs.calculator_inputs,
            label=self.metadata.label,
        )
        inputs["remote_workdir"] = self.inputs.calculator_inputs["remote_workdir"]
        VaspImmigrant = WorkflowFactory("vasp.immigrant")
        future = self.submit(VaspImmigrant, **inputs)
        self.report("{} pk = {}".format(self.metadata.label, future.pk))
        self.to_context(**{"calc": future})

    def validate_imported_structure(self):
        """Validate imported supercell structure.

        Only VaspImmigrantWorkChain is supported.

        """
        self.report("validate imported supercell structures")
        supercell_ref = self.inputs.structure
        supercell_calc = get_structure_from_vasp_immigrant(self.ctx.calc)
        if not compare_structures(
            supercell_ref, supercell_calc, self.inputs.symmetry_tolerance.value
        ):
            return self.exit_codes.ERROR_STRUCTURE_VALIDATION

    def finalize(self):
        """Finalize force calculation."""
        outputs = self.ctx.calc.outputs
        self.report("create forces ArrayData")
        forces = _get_forces(outputs, self.ctx.plugin_name)
        if forces is None:
            return self.exit_codes.ERROR_NO_FORCES
        else:
            self.out("forces", forces)

        self.report("create energy ArrayData")
        energy = _get_energy(outputs, self.ctx.plugin_name)
        if energy is None:
            return self.exit_codes.ERROR_NO_ENERGY
        else:
            self.out("energy", energy)
