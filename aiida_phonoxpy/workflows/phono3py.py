"""WorkChan to run ph-ph calculation by phono3py and force calculators."""

from aiida.engine import if_, while_
from aiida.orm import ArrayData, Dict, Float, StructureData

from aiida_phonoxpy.common.utils import setup_phono3py_calculation
from aiida_phonoxpy.workflows.base import BasePhonopyWorkChain
from aiida_phonoxpy.workflows.phonopy import PhonopyImmigrantMixIn


class Phono3pyWorkChain(BasePhonopyWorkChain, PhonopyImmigrantMixIn):
    """Phono3py workchain."""

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)
        spec.input("phonon_force_sets", valid_type=ArrayData, required=False)
        spec.input("immigrant_calculation_folders", valid_type=Dict, required=False)

        spec.outline(
            cls.initialize,
            if_(cls.import_calculations_from_files)(
                cls.initialize_immigrant,
                while_(cls.continue_import)(
                    cls.read_force_calculations_from_files,
                ),
                if_(cls.is_nac)(
                    cls.read_nac_calculations_from_files,
                ),
            ).else_(
                cls.run_force_and_nac_calculations,
            ),
            cls.create_force_sets,
            if_(cls.is_nac)(cls.attach_nac_params),
            if_(cls.run_phonopy)(
                if_(cls.remote_phonopy)(
                    cls.run_phono3py_remote,
                    cls.collect_remote_data,
                ).else_(
                    cls.create_force_constants,
                    cls.run_phono3py_in_workchain,
                )
            ),
        )
        spec.output("fc3", valid_type=ArrayData, required=False)
        spec.output("fc2", valid_type=ArrayData, required=False)
        spec.output("phonon_supercell", valid_type=StructureData, required=False)
        spec.output("phonon_force_sets", valid_type=ArrayData, required=False)
        spec.output("phonon_supercell_forces", valid_type=ArrayData, required=False)
        spec.output("phonon_supercell_energy", valid_type=Float, required=False)
        spec.exit_code(
            1003,
            "ERROR_INCONSISTENT_IMMIGRANT_FORCES_FOLDERS",
            message=(
                "Number of supercell folders is different from number "
                "of expected supercells."
            ),
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
        if "displacement_dataset" in self.inputs:
            kwargs["dataset"] = self.inputs.displacement_dataset
        return_vals = setup_phono3py_calculation(
            self.inputs.settings,
            self.inputs.structure,
            self.inputs.symmetry_tolerance,
            **kwargs
        )

        for key in ("phonon_setting_info", "primitive", "supercell"):
            self.ctx[key] = return_vals[key]
            self.out(key, self.ctx[key])
        self.ctx.supercells = {}
        self.ctx.phonon_supercells = {}
        for key in return_vals:
            if "supercell_" in key and "phonon_" not in key:
                self.ctx.supercells[key] = return_vals[key]
            if "phonon_supercell_" in key:
                self.ctx.phonon_supercells[key] = return_vals[key]
        self.ctx.primitive = return_vals["primitive"]
        self.ctx.supercell = return_vals["supercell"]
        if "phonon_supercell" in return_vals:
            self.ctx.phonon_supercell = return_vals["phonon_supercell"]

        if self.inputs.subtract_residual_forces:
            digits = len(str(len(self.ctx.supercells)))
            label = "supercell_%s" % "0".zfill(digits)
            self.ctx.supercells[label] = return_vals["supercell"]
            digits = len(str(len(self.ctx.phonon_supercells)))
            label = "phonon_supercell_%s" % "0".zfill(digits)
            self.ctx.phonon_supercells[label] = return_vals["phonon_supercell"]

    def run_force_and_nac_calculations(self):
        """Run supercell force, phonon supercell force, and NAC params calculations.

        Supercell force calculations and NAC params calculation
        are submitted in this method to make them run in parallel.

        """
        if "force_sets" in self.inputs:
            self.report("skip force calculation.")
            self.ctx.force_sets = self.inputs.force_sets
        else:
            self._run_force_calculations(self.ctx.supercells, label_prefix="force_calc")

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

    def postprocess_of_dry_run(self):
        """Do nothing."""
        self.report("Finish here because of dry-run setting")

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
