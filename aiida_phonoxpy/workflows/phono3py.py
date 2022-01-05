"""WorkChan to run ph-ph calculation by phono3py and force calculators."""

from aiida.engine import if_, while_
from aiida.orm import ArrayData, Dict, Float, StructureData

from aiida_phonoxpy.utils.utils import setup_phono3py_calculation
from aiida_phonoxpy.workflows.base import BasePhonopyWorkChain
from aiida_phonoxpy.workflows.phonopy import ImmigrantMixIn


class Phono3pyWorkChain(BasePhonopyWorkChain, ImmigrantMixIn):
    """Phono3py workchain.

    This workchain generates sudpercells with displacements and calculates
    supercell forces and parameters for non-analytical term correction.

    """

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)
        spec.input_namespace(
            "remote_workdirs",
            help="Directory names to import force and NAC calculations.",
        )
        spec.input(
            "remote_workdirs.force", valid_type=list, required=False, non_db=True
        )
        spec.input(
            "calculator_inputs.phonon_force",
            valid_type=dict,
            required=False,
            non_db=True,
        )
        spec.input("remote_workdirs.nac", valid_type=list, required=False, non_db=True)
        spec.input("phonon_force_sets", valid_type=ArrayData, required=False)
        spec.input("phonon_displacement_dataset", valid_type=Dict, required=False)
        spec.input("phonon_displacements", valid_type=ArrayData, required=False)

        spec.outline(
            cls.initialize,
            if_(cls.import_calculations_from_files)(
                cls.initialize_immigrant,
                while_(cls.continue_import)(
                    cls.import_force_calculations_from_files,
                ),
                if_(cls.is_nac)(
                    cls.import_nac_calculations_from_files,
                ),
            ).else_(
                cls.run_force_and_nac_calculations,
            ),
            cls.create_force_sets,
            if_(cls.should_run_phonon_supercell)(cls.create_phonon_force_sets),
            if_(cls.is_nac)(cls.attach_nac_params),
        )
        spec.output("fc3", valid_type=ArrayData, required=False)
        spec.output("fc2", valid_type=ArrayData, required=False)
        spec.output("phonon_supercell", valid_type=StructureData, required=False)
        spec.output("phonon_force_sets", valid_type=ArrayData, required=False)
        spec.output("phonon_supercell_forces", valid_type=ArrayData, required=False)
        spec.output("phonon_supercell_energy", valid_type=Float, required=False)
        spec.output("phonon_displacement_dataset", valid_type=Dict, required=False)
        spec.output("phonon_displacements", valid_type=ArrayData, required=False)
        spec.output(
            "phonon_supercell_forces",
            valid_type=ArrayData,
            required=False,
            help="Forces of perfect fc2 supercell.",
        )
        spec.output(
            "phonon_supercell_energy",
            valid_type=Float,
            required=False,
            help="Energy of perfect fc2 supercell.",
        )
        spec.exit_code(
            1003,
            "ERROR_INCONSISTENT_IMMIGRANT_FORCES_FOLDERS",
            message=(
                "Number of supercell folders is different from number "
                "of expected supercells."
            ),
        )

    def should_run_phonon_supercell(self):
        """Return boolen for outline."""
        return (
            "phonon_supercell_matrix" in self.inputs.settings.keys()
            and "phonon_force" in self.inputs.calculator_inputs
        )

    def continue_import(self):
        """Return boolen for outline."""
        total_num = self.ctx.num_supercell_forces + self.ctx.num_phonon_supercell_forces
        return self.ctx.num_imported < total_num

    def initialize_immigrant(self):
        """Initialize immigrant numbers."""
        self.ctx.num_imported = 0
        self.ctx.num_supercell_forces = len(
            self.inputs.immigrant_calculation_folders["forces"]
        )
        self.ctx.num_phonon_supercell_forces = len(
            self.inputs.immigrant_calculation_folders["phonon_forces"]
        )
        self.ctx.supercell_keys_done = []

        if len(self.ctx.supercells) != self.ctx.num_supercell_forces:
            return self.exit_codes.ERROR_INCONSISTENT_IMMIGRANT_FOLDERS
        if len(self.ctx.phonon_supercells) != self.ctx.num_phonon_supercell_forces:
            return self.exit_codes.ERROR_INCONSISTENT_IMMIGRANT_FOLDERS

    def initialize(self):
        """Set default settings and create supercells and primitive cell."""
        self.report("initialize")

        kwargs = {}
        for key in (
            "displacement_dataset",
            "displacements",
            "phonon_displacement_dataset",
            "phonon_displacements",
        ):
            if key in self.inputs:
                kwargs[key] = self.inputs[key]
                self.ctx[key] = self.inputs[key]
        return_vals = setup_phono3py_calculation(
            self.inputs.settings,
            self.inputs.structure,
            self.inputs.symmetry_tolerance,
            **kwargs,
        )

        for key in (
            "phonon_setting_info",
            "primitive",
            "supercell",
            "phonon_supercell",
            "displacements",
            "displacement_dataset",
            "phonon_displacements",
            "phonon_displacement_dataset",
        ):
            if key in return_vals:
                self.ctx[key] = return_vals[key]
                self.out(key, self.ctx[key])
        self.ctx.supercells = {}
        self.ctx.phonon_supercells = {}
        for key in return_vals:
            if "phonon_supercell_" in key:
                self.ctx.phonon_supercells[key] = return_vals[key]
            elif "supercell_" in key:
                self.ctx.supercells[key] = return_vals[key]

        if self.inputs.subtract_residual_forces:
            digits = len(str(len(self.ctx.supercells)))
            key = "supercell_%s" % "0".zfill(digits)
            self.ctx.supercells[key] = return_vals["supercell"]
            digits = len(str(len(self.ctx.phonon_supercells)))
            key = "phonon_supercell_%s" % "0".zfill(digits)
            self.ctx.phonon_supercells[key] = return_vals["phonon_supercell"]

    def run_force_and_nac_calculations(self):
        """Run supercell force, phonon supercell force, and NAC params calculations.

        Supercell force calculations and NAC params calculation
        are submitted in this method to make them run in parallel.

        """
        if "force_sets" in self.inputs:
            self.report("skip force calculation.")
            self.ctx.force_sets = self.inputs.force_sets
        else:
            self._run_force_calculations(self.ctx.supercells)

        if "phonon_supercell" in self.ctx:
            if "phonon_force_sets" in self.inputs:
                self.report("skip phonon force calculation.")
                self.ctx.phonon_force_sets = self.inputs.phonon_force_sets
            else:
                self._run_force_calculations(
                    self.ctx.phonon_supercells, label_prefix="phonon_force_calc"
                )

        if "nac_params" in self.inputs:
            self.report("skip nac params calculation.")
            self.ctx.nac_params = self.inputs.nac_params
        elif self.is_nac():
            self._run_nac_params_calculation()

    def create_phonon_force_sets(self):
        """Attach phonon force sets to outputs.

        outputs.phonon_force_sets
        outputs.phonon_supercell_forces (optional)
        outputs.phonon_supercell_energy (optional)

        """
        self.report("create phonon force sets")
        self._create_force_sets(self.ctx.phonon_supercells, key_prefix="phonon_")

    def run_phono3py_remote(self):
        """Do nothing."""
        self.report("remote phonopy calculation")

    def collect_remote_data(self):
        """Do nothing."""
        self.report("collect data")

    def create_force_constants(self):
        """Do nothing."""
        self.report("create force constants")

    def run_phono3py_in_workchain(self):
        """Do nothing."""
        self.report("phonopy calculation in workchain")
