"""WorkChain mix-in's."""
from aiida.orm import Int


class DoNothingMixIn:
    """Mix-in for DoNothingCalculation.

    Usage
    -----
    Insert following just before submit calculation in outline:

        if_(cls.use_donothing)(
            cls.do_nothing,
        ),

    """

    def use_donothing(self):
        """Return boolean for outline."""
        return "donothing_inputs" in self.inputs

    def do_nothing(self):
        """Sleep if inputs.seconds is given."""
        self.report("sleep")
        builder = self.inputs.donothing_inputs["code"].get_builder()
        builder.metadata.label = self.metadata.label
        if "metadata" in self.inputs.donothing_inputs:
            if "options" in self.inputs.donothing_inputs["metadata"]:
                options = self.inputs.donothing_inputs["metadata"]["options"]
                builder.metadata.options.update(options)
        if "seconds" in self.inputs.donothing_inputs:
            seconds = self.inputs.donothing_inputs["seconds"]
            if isinstance(seconds, Int):
                builder.seconds = seconds
            else:
                builder.seconds = Int(seconds)
        future = self.submit(builder)
        self.to_context(**{"donothing": future})


class ImmigrantMixIn:
    """List of methods to import calculations."""

    def import_calculations_from_files(self):
        """Return boolen for outline."""
        return "force" in self.inputs.remote_workdirs

    def import_force_calculations_from_files(self):
        """Import supercell force calculations.

        Importing backend works only for VASP.

        """
        from aiida_phonoxpy.workflows.forces import ForcesWorkChain

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
        from aiida_phonoxpy.workflows.nac_params import NacParamsWorkChain

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
