"""Workflow to calculate NAC params."""
from aiida.engine import WorkChain, append_, calcfunction, if_, while_
from aiida.orm import ArrayData, Float, StructureData
from aiida.plugins import WorkflowFactory
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon

from aiida_phonoxpy.common.builders import (
    get_calculator_process,
    get_import_workchain_inputs,
    get_plugin_names,
    get_workchain_inputs,
)
from aiida_phonoxpy.utils.utils import (
    compare_structures,
    get_structure_from_vasp_immigrant,
    phonopy_atoms_from_structure,
)
from aiida_phonoxpy.workflows.mixin import DoNothingMixIn


def _get_nac_params(ctx, symmetry_tolerance):
    """Obtain Born effective charges and dielectric constants in primitive cell.

    When Born effective charges and dielectric constants are calculated within
    phonopy workchain, those values are calculated in the primitive cell.
    However using immigrant, the cell may not be primitive cell and can be
    unit cell. In this case, conversion of data is necessary. This conversion
    needs information of the structure where those values were calcualted and
    the target primitive cell structure.

    When using immigrant, structure is in the immigrant calculation but not
    the workchain. 'structure' should be accessible in the vasp immigrant
    workchain level, and this should be fixed in aiida-vasp.

    """
    if ctx.plugin_names[0] == "vasp.vasp":
        calc = ctx.nac_params_calcs[0]
        if "structure" in calc.inputs:
            structure = calc.inputs.structure
        else:
            structure = get_structure_from_vasp_immigrant(calc)
        nac_params = get_vasp_nac_params(
            calc.outputs.born_charges,
            calc.outputs.dielectrics,
            structure,
            symmetry_tolerance,
        )
    elif ctx.plugin_names[0] == "quantumespresso.pw":
        pw_calc = ctx.nac_params_calcs[0]
        ph_calc = ctx.nac_params_calcs[1]
        nac_params = get_qe_nac_params(
            ph_calc.outputs.output_parameters,
            pw_calc.inputs.pw.structure,
            symmetry_tolerance,
        )
    else:
        nac_params = None
    return nac_params


@calcfunction
def get_qe_nac_params(output_parameters, structure, symmetry_tolerance, primitive=None):
    """Return NAC params ArrayData created from QE results."""
    nac_params = _get_nac_params_array(
        output_parameters["effective_charges_eu"],
        output_parameters["dielectric_constant"],
        structure,
        symmetry_tolerance.value,
        primitive=primitive,
    )
    return nac_params


@calcfunction
def get_vasp_nac_params(
    born_charges, epsilon, structure, symmetry_tolerance, primitive=None
):
    """Return NAC params ArrayData created from VASP results."""
    nac_params = _get_nac_params_array(
        born_charges.get_array("born_charges"),
        epsilon.get_array("epsilon"),
        structure,
        symmetry_tolerance.value,
        primitive=primitive,
    )
    return nac_params


def _get_nac_params_array(
    born_charges, epsilon, structure, symmetry_tolerance, primitive=None
):
    phonopy_cell = phonopy_atoms_from_structure(structure)
    if primitive is not None:
        phonopy_primitive = phonopy_atoms_from_structure(primitive)
    else:
        phonopy_primitive = None
    borns_, epsilon_ = symmetrize_borns_and_epsilon(
        born_charges,
        epsilon,
        phonopy_cell,
        symprec=symmetry_tolerance,
        primitive=phonopy_primitive,
    )
    nac_params = ArrayData()
    nac_params.set_array("born_charges", borns_)
    nac_params.set_array("epsilon", epsilon_)
    nac_params.label = "born_charges & epsilon"
    return nac_params


class NacParamsWorkChain(WorkChain, DoNothingMixIn):
    """Wrapper to compute non-analytical term correction parameters."""

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)
        spec.input("structure", valid_type=StructureData, required=True)
        spec.input("calculator_inputs", valid_type=dict, required=True, non_db=True)
        spec.input("symmetry_tolerance", valid_type=Float, default=lambda: Float(1e-5))
        spec.input("donothing_inputs", valid_type=dict, required=False, non_db=True)

        spec.outline(
            cls.initialize,
            while_(cls.continue_calculation)(
                if_(cls.import_calculation)(
                    cls.import_calculation_from_workdir,
                    cls.validate_imported_structure,
                ).else_(
                    if_(cls.use_donothing)(
                        cls.do_nothing,
                    ),
                    cls.run_calculation,
                ),
            ),
            cls.finalize,
        )

        spec.output("nac_params", valid_type=ArrayData, required=True)

        spec.exit_code(
            1001,
            "ERROR_NO_NAC_PARAMS",
            message=("NAC params could not be retrieved from calculaton."),
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
        """Return boolen for outline."""
        return "remote_workdir" in self.inputs.calculator_inputs

    def continue_calculation(self):
        """Return boolen for outline."""
        if self.ctx.iteration >= self.ctx.max_iteration:
            return False
        self.ctx.iteration += 1
        return True

    def import_calculation_from_files(self):
        """Return boolen for outline."""
        return "immigrant_calculation_folder" in self.inputs

    def initialize(self):
        """Initialize outline control parameters."""
        self.report("initialization")
        self.ctx.iteration = 0
        if "steps" in self.inputs.calculator_inputs.keys():
            self.ctx.max_iteration = len(self.inputs.calculator_inputs["steps"])
        else:
            self.ctx.max_iteration = 1

        self.ctx.plugin_names = get_plugin_names(self.inputs.calculator_inputs)

    def run_calculation(self):
        """Run NAC params calculation."""
        self.report(
            "calculation iteration %d/%d" % (self.ctx.iteration, self.ctx.max_iteration)
        )

        if "steps" in self.inputs.calculator_inputs.keys():
            calculator_inputs = self.inputs.calculator_inputs["steps"][
                self.ctx.iteration - 1
            ]
        else:
            calculator_inputs = self.inputs.calculator_inputs
        label = "nac_params_%d" % self.ctx.iteration

        process_inputs = get_workchain_inputs(
            calculator_inputs,
            self.inputs.structure,
            ctx=self.ctx,
            label=label,
        )
        i = self.ctx.iteration - 1
        CalculatorProcess = get_calculator_process(plugin_name=self.ctx.plugin_names[i])
        future = self.submit(CalculatorProcess, **process_inputs)
        self.report("nac_params: {}".format(future.pk))
        self.to_context(nac_params_calcs=append_(future))

    def import_calculation_from_workdir(self):
        """Import NAC parameter calculation.

        Only VaspImmigrantWorkChain is supported.

        """
        self.report(
            "import calculation data in files %d/%d"
            % (self.ctx.iteration, self.ctx.max_iteration)
        )
        label = "nac_params_%d" % self.ctx.iteration
        inputs = get_import_workchain_inputs(self.inputs.calculator_inputs, label=label)
        inputs["remote_workdir"] = self.inputs.calculator_inputs["remote_workdir"]
        VaspImmigrant = WorkflowFactory("vasp.immigrant")
        future = self.submit(VaspImmigrant, **inputs)
        self.report("nac_params: {}".format(future.pk))
        self.to_context(nac_params_calcs=append_(future))

    def validate_imported_structure(self):
        """Validate imported structure.

        Only VaspImmigrantWorkChain is supported.

        """
        self.report("validate imported structures")
        supercell_ref = self.inputs.structure
        supercell_calc = get_structure_from_vasp_immigrant(self.ctx.nac_params_calcs[0])
        if not compare_structures(
            supercell_ref, supercell_calc, self.inputs.symmetry_tolerance.value
        ):
            return self.exit_codes.ERROR_STRUCTURE_VALIDATION

    def finalize(self):
        """Finalize NAC params calculation."""
        self.report("finalization")

        nac_params = _get_nac_params(self.ctx, self.inputs.symmetry_tolerance)
        if nac_params is None:
            return self.exit_codes.ERROR_NO_NAC_PARAMS

        self.out("nac_params", nac_params)
