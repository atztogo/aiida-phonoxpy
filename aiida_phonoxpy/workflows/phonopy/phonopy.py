"""PhonopyWorkChain."""

from aiida.engine import if_, while_
from aiida.plugins import DataFactory

from aiida_phonoxpy.workflows.forces import ForcesWorkChain
from aiida_phonoxpy.workflows.nac_params import NacParamsWorkChain
from aiida_phonoxpy.workflows.phonopy.base import BasePhonopyWorkChain

Dict = DataFactory("dict")
Str = DataFactory("str")


class PhonopyImmigrantMixIn:
    """List of methods to import calculations."""

    def import_calculations_from_files(self):
        """Return boolen for outline."""
        return "force" in self.inputs.remote_workdirs

    def import_force_calculations_from_files(self):
        """Import supercell force calculations.

        Importing backend works only for VASP.

        """
        self.report("import supercell force calculation data in files.")
        num_batch = 50
        self.report("%d calculations per batch." % num_batch)

        calc_folders = self.inputs.remote_workdirs.force
        local_count = 0
        for key, supercell in self.ctx.supercells.items():
            if key in self.ctx.supercell_keys_done:
                continue
            else:
                self.ctx.supercell_keys_done.append(key)
            label = "force_calc_%s" % key.split("_")[-1]
            number = int(key.split("_")[-1])
            if not self.inputs.subtract_residual_forces:
                number -= 1
            builder = ForcesWorkChain.get_builder()
            builder.metadata.label = label
            builder.structure = supercell
            calculator_inputs = {"remote_workdir": calc_folders[number]}
            if "force" in self.inputs.calculator_inputs:
                calculator_inputs.update(self.inputs.calculator_inputs.force)
            else:
                calculator_inputs.update(self.inputs.calculator_settings["forces"])
                self.logger.warning(
                    "Use calculator_inputs.force instead of "
                    "calculator_settings['forces']."
                )
            builder.calculator_inputs = calculator_inputs
            future = self.submit(builder)
            self.report("{} pk = {}".format(label, future.pk))
            self.to_context(**{label: future})
            self.ctx.num_imported += 1
            if not self.continue_import():
                break
            local_count += 1
            if local_count == num_batch:
                break

    def import_nac_calculations_from_files(self):
        """Import NAC params calculation.

        Importing backend works only for VASP.

        """
        self.report("import NAC calculation data in files")
        label = "nac_params_calc"
        builder = NacParamsWorkChain.get_builder()
        builder.metadata.label = label
        builder.structure = self.ctx.primitive
        calculator_inputs = {"remote_workdir": self.inputs.remote_workdirs.nac[0]}
        if "nac" in self.inputs.calculator_inputs:
            calculator_inputs.update(self.inputs.calculator_inputs.nac)
        else:
            calculator_inputs.update(self.inputs.calculator_settings["nac"])
            self.logger.warning(
                "Use calculator_inputs.force instead of "
                "calculator_settings['forces']."
            )
        builder.calculator_inputs = calculator_inputs
        future = self.submit(builder)
        self.report("{} pk = {}".format(label, future.pk))
        self.to_context(**{label: future})


class PhonopyWorkChain(BasePhonopyWorkChain, PhonopyImmigrantMixIn):
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
            if_(cls.force_sets_exists)(cls.do_nothing).else_(
                cls.create_force_sets,
            ),
            if_(cls.nac_params_exists)(cls.do_nothing).else_(
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
