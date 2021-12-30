"""PhonopyWorkChain."""

from aiida.engine import if_, while_
from aiida.orm import Dict

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
        spec.input_namespace(
            "remote_workdirs",
            help="Directory names to import force and NAC calculations.",
        )
        spec.input(
            "remote_workdirs.force", valid_type=list, required=False, non_db=True
        )
        spec.input("remote_workdirs.nac", valid_type=list, required=False, non_db=True)
        spec.input("calculator_settings", valid_type=Dict, required=False)

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
            if_(cls.force_sets_exists)(cls.do_pass).else_(
                cls.create_force_sets,
            ),
            if_(cls.nac_params_exists)(cls.do_pass).else_(
                if_(cls.is_nac)(cls.attach_nac_params),
            ),
            if_(cls.run_phonopy)(
                if_(cls.remote_phonopy)(
                    cls.run_phonopy_remote,
                    cls.collect_remote_data,
                ).else_(
                    cls.create_force_constants,
                    cls.run_phonopy_locally,
                ),
            ),
            cls.finalize,
        )
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
        return self.ctx.num_imported < self.ctx.num_supercell_forces

    def initialize_immigrant(self):
        """Initialize immigrant numbers."""
        self.ctx.num_imported = 0
        self.ctx.num_supercell_forces = len(self.inputs.remote_workdirs.force)
        self.ctx.supercell_keys_done = []

        if len(self.ctx.supercells) != self.ctx.num_supercell_forces:
            return self.exit_codes.ERROR_INCONSISTENT_IMMIGRANT_FOLDERS

    def is_nac(self):
        """Return boolean for outline."""
        if "nac" in self.inputs.remote_workdirs:
            return True
        return super().is_nac()
