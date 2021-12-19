"""Workflow to calculate supercell forces."""
import time

import numpy as np
from aiida.engine import WorkChain, calcfunction, if_
from aiida.orm import Group, QueryBuilder, WorkChainNode, load_group
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_phonopy.common.builders import (
    get_calculator_process,
    get_plugin_names,
    get_vasp_immigrant_inputs,
    get_workchain_inputs,
)
from aiida_phonopy.common.utils import (
    compare_structures,
    get_structure_from_vasp_immigrant,
)

Float = DataFactory("float")
Dict = DataFactory("dict")
Str = DataFactory("str")
ArrayData = DataFactory("array")
StructureData = DataFactory("structure")


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
            energy_data = get_vasp_energy(outputs.energies)
        else:
            return None
    elif plugin_name == "quantumespresso.pw":
        if (
            "output_parameters" in outputs
            and "energy" in outputs.output_parameters.keys()
        ):
            energy_data = get_qe_energy(outputs.output_parameters)
    return energy_data


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


class ForcesWorkChain(WorkChain):
    """Wrapper to compute supercell forces."""

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)
        spec.input("structure", valid_type=StructureData, required=True)
        spec.input("calculator_inputs", valid_type=dict, required=True, non_db=True)
        spec.input("symmetry_tolerance", valid_type=Float, default=lambda: Float(1e-5))
        spec.input("immigrant_calculation_folder", valid_type=Str, required=False)
        spec.input("queue_name", valid_type=Str, required=False)
        spec.outline(
            cls.initialize,
            if_(cls.import_calculation_from_files)(
                cls.read_calculation_from_folder,
                cls.validate_imported_structure,
            ).else_(
                if_(cls.use_queue)(cls.submit_to_queue),
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

    def import_calculation_from_files(self):
        """Return boolen for outline."""
        return "immigrant_calculation_folder" in self.inputs

    def use_queue(self):
        """Use queue to wait for submitting calculation."""
        return "queue_name" in self.inputs

    def initialize(self):
        """Initialize outline control parameters."""
        self.report("initialization")
        self.ctx.plugin_name = get_plugin_names(self.inputs.calculator_inputs)[0]

    def submit_to_queue(self):
        """Wait until being ready to submit calculation."""
        self.report("waiting")
        g = load_group(self.inputs.queue_name.value + "/submit")
        g.add_nodes(self.node)
        for i in range(1000):
            qb = QueryBuilder()
            qb.append(Group, filters={"label": "queue/run"}, tag="group")
            qb.append(
                WorkChainNode, with_group="group", filters={"uuid": self.node.uuid}
            )
            if qb.count() > 0:
                return
            time.sleep(10)
        return self.exit_codes.ERROR_WAITING_TIMEOUT

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

    def read_calculation_from_folder(self):
        """Import supercell force calculation using immigrant."""
        self.report("import supercell force calculation data in files.")
        force_folder = self.inputs.immigrant_calculation_folder
        inputs = get_vasp_immigrant_inputs(
            force_folder.value,
            self.inputs.calculator_inputs.dict,
            label=self.metadata.label,
        )
        VaspImmigrant = WorkflowFactory("vasp.immigrant")
        future = self.submit(VaspImmigrant, **inputs)
        self.report("{} pk = {}".format(self.metadata.label, future.pk))
        self.to_context(**{"calc": future})

    def validate_imported_structure(self):
        """Validate imported supercell structure."""
        self.report("validate imported supercell structures")
        supercell_ref = self.inputs.structure
        supercell_calc = get_structure_from_vasp_immigrant(self.ctx.calc)
        if not compare_structures(
            supercell_ref, supercell_calc, self.inputs.symmetry_tolerance
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
