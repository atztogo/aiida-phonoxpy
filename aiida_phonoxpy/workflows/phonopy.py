"""PhonopyWorkChain."""

from aiida.engine import if_, while_
from aiida.orm import BandsData, Bool, Code, Dict, XyData

from aiida_phonoxpy.calculations.phonopy import PhonopyCalculation
from aiida_phonoxpy.utils.utils import (
    get_force_constants,
    get_phonon_properties,
    setup_phonopy_calculation,
)
from aiida_phonoxpy.workflows.base import BasePhonopyWorkChain
from aiida_phonoxpy.workflows.mixin import ImmigrantMixIn


class PhonopyWorkChain(BasePhonopyWorkChain, ImmigrantMixIn):
    """Phonopy workchain.

    inputs
    ------
    See most of inputs at BasePhonopyWorkChain.
    calculator_settings : Dict
        Deprecated.
        Settings to run force and nac calculations. For example,
            {'forces': force_config,
             'nac': nac_config}
        At least 'forces' key is necessary. 'nac' is optional.
        force_config is used for supercell force calculation. nac_config
        are used for Born effective charges and dielectric constant calculation
        in primitive cell. The primitive cell is chosen by phonopy
        automatically.
    immigrant_calculation_folders : Dict, optional
        'force' key has to exist and 'nac' is necessary when
        phonon_settings['is_nac'] is True. The value of the 'force' key is
        the list of strings of remote directories. The value of 'nac' is the
        string of remote directory.

    """

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)
        spec.expose_inputs(
            PhonopyCalculation, namespace="phonopy", include=("metadata",)
        )
        spec.input(
            "phonopy.metadata.options.resources", valid_type=dict, required=False
        )
        spec.input_namespace(
            "remote_workdirs",
            help="Directory names to import force and NAC calculations.",
        )
        spec.input(
            "remote_workdirs.force", valid_type=list, required=False, non_db=True
        )
        spec.input("remote_workdirs.nac", valid_type=list, required=False, non_db=True)
        spec.input("calculator_settings", valid_type=Dict, required=False)
        spec.input("run_phonopy", valid_type=Bool, default=lambda: Bool(False))
        spec.input("remote_phonopy", valid_type=Bool, default=lambda: Bool(True))

        spec.outline(
            cls.initialize,
            if_(cls.import_calculations_from_files)(
                cls.initialize_immigrant,
                if_(cls.is_force)(
                    while_(cls.continue_import)(
                        cls.import_force_calculations_from_files,
                    ),
                ),
                if_(cls.is_nac)(
                    cls.import_nac_calculations_from_files,
                ),
            ).else_(
                cls.run_force_and_nac_calculations,
            ),
            if_(cls.force_sets_exists)(cls.do_pass).else_(
                if_(cls.is_force)(
                    cls.create_force_sets,
                ),
            ),
            if_(cls.nac_params_exists)(cls.do_pass).else_(
                if_(cls.is_nac)(cls.attach_nac_params),
            ),
            if_(cls.should_run_phonopy)(
                if_(cls.is_force)(
                    if_(cls.should_run_remote_phonopy)(
                        cls.run_phonopy_remote,
                        cls.collect_remote_data,
                    ).else_(
                        cls.create_force_constants,
                        cls.run_phonopy_locally,
                    ),
                ),
            ),
            cls.finalize,
        )
        spec.output("thermal_properties", valid_type=XyData, required=False)
        spec.output("band_structure", valid_type=BandsData, required=False)
        spec.output("total_dos", valid_type=XyData, required=False)
        spec.output("projected_dos", valid_type=XyData, required=False)
        spec.exit_code(
            1003,
            "ERROR_INCONSISTENT_IMMIGRANT_FORCES_FOLDERS",
            message=(
                "Number of supercell folders is different from number "
                "of expected supercells."
            ),
        )

    def should_run_remote_phonopy(self):
        """Return boolean for outline."""
        return self.inputs.remote_phonopy

    def should_run_phonopy(self):
        """Return boolean for outline."""
        return self.inputs.run_phonopy

    def continue_import(self):
        """Return boolen for outline."""
        return self.ctx.num_imported < self.ctx.num_supercell_forces

    def initialize(self):
        """Set default settings and create supercells and primitive cell.

        self.ctx.supercells contains supercells as a dict.
        The keys are like 'spercell_000', 'supercell_001', ...,
        where the number of digits depends on the number of supercells.
        'spercell_000' is only available when
            self.inputs.subtract_residual_forces = True.

        """
        self.report("initialization")

        if self.inputs.run_phonopy and self.inputs.remote_phonopy:
            if "code" not in self.inputs and "code_string" not in self.inputs:
                return self.exit_codes.ERROR_NO_PHONOPY_CODE

        if "supercell_matrix" not in self.inputs.settings.keys():
            return self.exit_codes.ERROR_NO_SUPERCELL_MATRIX

        kwargs = {}
        for key in ("displacement_dataset", "displacements"):
            if key in self.inputs:
                kwargs[key] = self.inputs[key]
                self.ctx[key] = self.inputs[key]
        return_vals = setup_phonopy_calculation(
            self.inputs.settings,
            self.inputs.structure,
            self.inputs.symmetry_tolerance,
            self.inputs.run_phonopy,
            **kwargs,
        )

        for key in (
            "phonon_setting_info",
            "primitive",
            "supercell",
            "displacements",
            "displacement_dataset",
        ):
            if key in return_vals:
                self.ctx[key] = return_vals[key]
                self.out(key, self.ctx[key])
        self.ctx.supercells = {}
        if self.inputs.subtract_residual_forces:
            digits = len(str(len(self.ctx.supercells)))
            key = "supercell_%s" % "0".zfill(digits)
            self.ctx.supercells[key] = return_vals["supercell"]
        for key in return_vals:
            if "supercell_" in key:
                self.ctx.supercells[key] = return_vals[key]

    def initialize_immigrant(self):
        """Initialize immigrant numbers."""
        self.ctx.num_imported = 0
        self.ctx.num_supercell_forces = len(self.inputs.remote_workdirs.force)
        self.ctx.supercell_keys_done = []

        if len(self.ctx.supercells) != self.ctx.num_supercell_forces:
            return self.exit_codes.ERROR_INCONSISTENT_IMMIGRANT_FOLDERS

    def run_force_and_nac_calculations(self):
        """Run supercell force and NAC params calculations.

        Supercell force calculations and NAC params calculation
        are submitted in this method to make them run in parallel.

        """
        if "force_sets" in self.inputs:
            self.report("skip force calculation.")
            self.ctx.force_sets = self.inputs.force_sets
        elif self.is_force():
            self._run_force_calculations(self.ctx.supercells)

        if "nac_params" in self.inputs:
            self.report("skip nac params calculation.")
            self.ctx.nac_params = self.inputs.nac_params
        elif self.is_nac():
            self._run_nac_params_calculation()

    def run_phonopy_remote(self):
        """Run phonopy at remote computer."""
        self.report("remote phonopy calculation")

        if "code_string" in self.inputs:
            code = Code.get_from_string(self.inputs.code_string.value)
        elif "code" in self.inputs:
            code = self.inputs.code
        builder = code.get_builder()
        builder.structure = self.inputs.structure
        builder.settings = self.ctx.phonon_setting_info
        builder.symmetry_tolerance = self.inputs.symmetry_tolerance
        if "label" in self.inputs.metadata:
            builder.metadata.label = self.inputs.metadata.label
        if "options" in self.inputs.phonopy.metadata:
            builder.metadata.options.update(self.inputs.phonopy.metadata.options)
        builder.force_sets = self.ctx.force_sets
        if "nac_params" in self.ctx:
            builder.nac_params = self.ctx.nac_params
        if "displacements" in self.ctx:
            builder.displacements = self.ctx.displacements
        if "displacement_dataset" in self.ctx:
            builder.displacement_dataset = self.ctx.displacement_dataset
        future = self.submit(builder)

        self.report("phonopy calculation: {}".format(future.pk))
        self.to_context(**{"phonon_properties": future})
        # return ToContext(phonon_properties=future)

    def collect_remote_data(self):
        """Collect phonon data from remove phonopy calculation."""
        self.report("collect data")
        ph_props = (
            "thermal_properties",
            "total_dos",
            "projected_dos",
            "band_structure",
            "force_constants",
        )

        for prop in ph_props:
            if prop in self.ctx.phonon_properties.outputs:
                self.out(prop, self.ctx.phonon_properties.outputs[prop])

        self.report("finish phonon")

    def create_force_constants(self):
        """Create force constants for run_phonopy_locally."""
        self.report("create force constants")

        kwargs = {
            key: self.ctx[key]
            for key in ("displacement_dataset", "displacements")
            if key in self.ctx
        }
        self.ctx.force_constants = get_force_constants(
            self.inputs.structure,
            self.ctx.phonon_setting_info,
            self.ctx.force_sets,
            self.inputs.symmetry_tolerance,
            **kwargs,
        )
        self.out("force_constants", self.ctx.force_constants)

    def run_phonopy_locally(self):
        """Run phonopy calculation locally."""
        self.report("phonopy calculation in workchain")

        nac_params = None
        if "nac_params" in self.ctx:
            nac_params = self.ctx.nac_params
        result = get_phonon_properties(
            self.inputs.structure,
            self.ctx.phonon_setting_info,
            self.ctx.force_constants,
            nac_params=nac_params,
        )
        self.out("thermal_properties", result["thermal_properties"])
        self.out("total_dos", result["total_dos"])
        self.out("band_structure", result["band_structure"])

        self.report("finish phonon")

    def finalize(self):
        """Show final message."""
        self.report("phonopy calculation has been done.")
