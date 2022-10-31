"""BasePhonopyWorkChain."""

from aiida.engine import WorkChain
from aiida.orm import ArrayData, Bool, Code, Dict, Float, Str, StructureData

from aiida_phonoxpy.utils.utils import collect_forces_and_energies, get_force_sets
from aiida_phonoxpy.workflows.forces import ForcesWorkChain
from aiida_phonoxpy.workflows.nac_params import NacParamsWorkChain


class BasePhonopyWorkChain(WorkChain):
    """BasePhonopyWorkchain.

    inputs
    ------
    structure : StructureData
        Unit cell structure.
    phonon_settings : Dict
        Setting to run phonon calculation. Keys are:
        supercell_matrix : list or list of list
            Multiplicity to create supercell from unit cell. Three integer
            values (list) or 3x3 integer values (list of list).
        mesh : list of float, optional
            List of three integer values or float to represent distance between
            neighboring q-points. Default is 100.0.
        distance : float, optional
            Atomic displacement distance. Default is 0.01.
        is_nac : bool, optional
            Deprecated.
            Whether running non-analytical term correction or not. Default is
            False.
        fc_calculator : str
            With this being 'alm', ALM is used to calculate force constants in
            the remote phonopy calculation.
        options : dict
            AiiDA calculation options for phonon calculation used when both of
            run_phonopy and remote_phonopy are True.
    displacement_dataset : Dict, optional
        Phonopy's type-1 displacement dataset. When force_sets is also given,
        this is used to compute force constants.
    displacements : ArrayData, optional
        Displacements of all atoms in supercells corresponding to force_sets.
        When force_sets is also given, this is used to compute force constants
        using ALM. Therefore, ALM has to be installed.
    force_sets : ArrayData, optional
        Supercell forces. When this is supplied, force calculation is skipped.
    nac_params : ArrayData, optional
        NAC parameters. When this is supplied, NAC calculation is skipped.
    calculator_inputs.force : dict, optional
        This is used for supercell force calculation.
    calculator_inputs.nac : dict, optional
        This is used for Born effective charges and dielectric constant calculation
        in primitive cell. The primitive cell is chosen by phonopy
        automatically.
    donothing_inputs : dict, optional
        This is used when donothing plugin to control submission of calculations.
    subtract_residual_forces : Bool, optional
        Run a perfect supercell force calculation and subtract the residual
        forces from forces in supercells with displacements. Default is False.
    run_phonopy : Bool, optional
        Whether running phonon calculation or not. Default is False.
    remote_phonopy : Bool, optional
        Whether running phonon calculation or not at remote. Default is False.
    code_string : Str, optional
        Code string of phonopy needed when both of run_phonopy and
        remote_phonopy are True.
    symmetry_tolerance : Float, optional
        Symmetry tolerance. Default is 1e-5.

    """

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)
        spec.input("structure", valid_type=StructureData, required=True)
        spec.input("settings", valid_type=Dict, default=lambda: Dict(dict={}))
        spec.input(
            "calculator_inputs.force", valid_type=dict, required=False, non_db=True
        )
        spec.input(
            "calculator_inputs.nac", valid_type=dict, required=False, non_db=True
        )
        spec.input("symmetry_tolerance", valid_type=Float, default=lambda: Float(1e-5))
        spec.input(
            "subtract_residual_forces", valid_type=Bool, default=lambda: Bool(False)
        )
        spec.input("nac_structure", valid_type=StructureData, required=False)
        spec.input("displacement_dataset", valid_type=Dict, required=False)
        spec.input("displacements", valid_type=ArrayData, required=False)
        spec.input("force_sets", valid_type=ArrayData, required=False)
        spec.input("nac_params", valid_type=ArrayData, required=False)
        spec.input("code_string", valid_type=Str, required=False)
        spec.input("code", valid_type=Code, required=False)
        spec.input("donothing_inputs", valid_type=dict, required=False, non_db=True)

        spec.output("primitive", valid_type=StructureData, required=False)
        spec.output("supercell", valid_type=StructureData, required=False)
        spec.output("displacements", valid_type=ArrayData, required=False)
        spec.output("displacement_dataset", valid_type=Dict, required=False)
        spec.output("force_sets", valid_type=ArrayData, required=False)
        spec.output(
            "supercell_forces",
            valid_type=ArrayData,
            required=False,
            help="Forces of perfect supercell.",
        )
        spec.output(
            "supercell_energy",
            valid_type=(Float, ArrayData),
            required=False,
            help="Energy of perfect supercell.",
        )
        spec.output("nac_params", valid_type=ArrayData, required=False)
        spec.output("phonon_setting_info", valid_type=Dict, required=False)
        spec.exit_code(
            1001,
            "ERROR_NO_PHONOPY_CODE",
            message=(
                "Phonopy Code not found though expected to run phonopy " "remotely."
            ),
        )
        spec.exit_code(
            1002,
            "ERROR_NO_SUPERCELL_MATRIX",
            message=("supercell_matrix was not found."),
        )

    def do_pass(self):
        """Do nothing."""
        return

    def is_nac(self):
        """Return boolean for outline."""
        if "nac" in self.inputs.calculator_inputs:
            return True
        if "is_nac" in self.inputs.settings.keys():
            self.logger.warning("Use inputs.settings['is_nac'] is deprecated.")
            return self.inputs.settings["is_nac"]
        return False

    def is_force(self):
        """Return boolean for outline."""
        if "supercell_matrix" not in self.inputs.settings:
            return False
        if "force" in self.inputs.calculator_inputs:
            return True
        return False

    def force_sets_exists(self):
        """Return boolean for outline."""
        return "force_sets" in self.inputs or "force_sets" in self.ctx

    def nac_params_exists(self):
        """Return boolean for outline."""
        return "nac_params" in self.inputs

    def _run_force_calculations(self, supercells, label_prefix="force_calc"):
        """Run supercell force calculations."""
        self.report(f"run force calculations ({label_prefix})")

        for key, supercell in supercells.items():
            num = key.split("_")[-1]
            label = f"{label_prefix}_{num}"
            builder = ForcesWorkChain.get_builder()
            builder.metadata.label = label
            builder.structure = supercell
            if "force" in self.inputs.calculator_inputs:
                if "phonon_supercell_" in key:
                    calculator_inputs = self.inputs.calculator_inputs.phonon_force
                else:
                    calculator_inputs = self.inputs.calculator_inputs.force
            else:
                calculator_inputs = self.inputs.calculator_settings["forces"]
                self.logger.warning(
                    "Use calculator_inputs.force instead of "
                    "calculator_settings['forces']."
                )
            builder.calculator_inputs = calculator_inputs
            if "donothing_inputs" in self.inputs:
                builder.donothing_inputs = self.inputs.donothing_inputs
            future = self.submit(builder)
            self.report("{} pk = {}".format(label, future.pk))
            self.to_context(**{label: future})

    def _run_nac_params_calculation(self):
        """Run nac params calculation."""
        self.report("run nac params calculation")

        builder = NacParamsWorkChain.get_builder()
        builder.metadata.label = "nac_params_calc"
        if "nac_structure" in self.inputs:
            builder.structure = self.inputs.nac_structure
            builder.primitive_structure = self.ctx.primitive
        else:
            builder.structure = self.ctx.primitive
        if "nac" in self.inputs.calculator_inputs:
            calculator_inputs = self.inputs.calculator_inputs.nac
        else:
            calculator_inputs = self.inputs.calculator_settings["nac"]
            self.logger.warning(
                "Use calculator_inputs.nac instead of calculator_settings['nac']."
            )

        builder.calculator_inputs = calculator_inputs
        if "donothing_inputs" in self.inputs:
            builder.donothing_inputs = self.inputs.donothing_inputs
        future = self.submit(builder)
        self.report("nac_params: {}".format(future.pk))
        self.to_context(**{"nac_params_calc": future})

    def create_force_sets(self):
        """Attach force sets to outputs.

        outputs.force_sets
        outputs.supercell_forces (optional)
        outputs.supercell_energy (optional)

        """
        self.report("create force sets")
        self._create_force_sets(self.ctx.supercells)

    def _create_force_sets(self, supercells, key_prefix=""):
        calc_key_prefix = key_prefix + "force_calc"
        forces_dict = collect_forces_and_energies(
            self.ctx, supercells, calc_key_prefix=calc_key_prefix
        )

        for key, val in get_force_sets(**forces_dict).items():
            out_key = key_prefix + key
            self.ctx[out_key] = val
            self.out(out_key, self.ctx[out_key])

        # when subtract_residual_forces = True
        for key in forces_dict:
            if int(key.split("_")[-1]) == 0:
                if "forces" in key:
                    self.out("supercell_forces", forces_dict[key])
                if "energy" in key:
                    self.out("supercell_energy", forces_dict[key])

    def attach_nac_params(self):
        """Attach nac_params ArrayData to outputs."""
        self.report("create nac params")

        self.ctx.nac_params = self.ctx.nac_params_calc.outputs.nac_params
        self.out("nac_params", self.ctx.nac_params)
