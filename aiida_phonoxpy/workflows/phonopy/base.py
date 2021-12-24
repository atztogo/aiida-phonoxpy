"""BasePhonopyWorkChain."""

from aiida.engine import WorkChain, if_
from aiida.orm import (
    Code,
    Float,
    Bool,
    Str,
    Dict,
    ArrayData,
    XyData,
    StructureData,
    BandsData,
)
from aiida_phonoxpy.common.utils import (
    collect_forces_and_energies,
    get_force_constants,
    get_force_sets,
    get_phonon_properties,
    setup_phonopy_calculation,
)
from aiida_phonoxpy.workflows.forces import ForcesWorkChain
from aiida_phonoxpy.workflows.nac_params import NacParamsWorkChain
from aiida_phonoxpy.calcs.phonopy import PhonopyCalculation


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
    queue_name : Str, optional
        When supplied, WorkChainNode is added to the group "<queue_name>/submit".
        Then this node entry is found in "<queue_name>/run", the WorkChainNode is
        submitted to aiida daemon. It is assumed that an external agent copies
        the node entry to "<queue_name>/run".

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
        spec.input("structure", valid_type=StructureData, required=True)
        spec.input("settings", valid_type=Dict, required=True)
        spec.input_namespace(
            "calculator_inputs", help="Inputs passed to force and NAC calculators."
        )
        spec.input(
            "calculator_inputs.force", valid_type=dict, required=False, non_db=True
        )
        spec.input(
            "calculator_inputs.nac", valid_type=dict, required=False, non_db=True
        )
        spec.input_namespace(
            "remote_workdirs",
            help="Directory names to import force and NAC calculations.",
        )
        spec.input(
            "remote_workdirs.force", valid_type=list, required=False, non_db=True
        )
        spec.input("remote_workdirs.nac", valid_type=list, required=False, non_db=True)
        spec.input("symmetry_tolerance", valid_type=Float, default=lambda: Float(1e-5))
        spec.input(
            "subtract_residual_forces", valid_type=Bool, default=lambda: Bool(False)
        )
        spec.input("run_phonopy", valid_type=Bool, default=lambda: Bool(False))
        spec.input("remote_phonopy", valid_type=Bool, default=lambda: Bool(False))
        spec.input("displacement_dataset", valid_type=Dict, required=False)
        spec.input("displacements", valid_type=ArrayData, required=False)
        spec.input("force_sets", valid_type=ArrayData, required=False)
        spec.input("nac_params", valid_type=ArrayData, required=False)
        spec.input("code_string", valid_type=Str, required=False)
        spec.input("code", valid_type=Code, required=False)
        spec.input("queue_name", valid_type=Str, required=False)

        spec.outline(
            cls.initialize,
            cls.run_force_and_nac_calculations,
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
        spec.output("force_constants", valid_type=ArrayData, required=False)
        spec.output("primitive", valid_type=StructureData, required=False)
        spec.output("supercell", valid_type=StructureData, required=False)
        spec.output("displacements", valid_type=ArrayData, required=False)
        spec.output("displacement_dataset", valid_type=Dict, required=False)
        spec.output("force_sets", valid_type=ArrayData, required=False)
        spec.output("supercell_forces", valid_type=ArrayData, required=False)
        spec.output("supercell_energy", valid_type=Float, required=False)
        spec.output("nac_params", valid_type=ArrayData, required=False)
        spec.output("thermal_properties", valid_type=XyData, required=False)
        spec.output("band_structure", valid_type=BandsData, required=False)
        spec.output("dos", valid_type=XyData, required=False)
        spec.output("pdos", valid_type=XyData, required=False)
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

    def do_nothing(self):
        """Do nothing."""
        return

    def remote_phonopy(self):
        """Return boolean for outline."""
        return self.inputs.remote_phonopy

    def run_phonopy(self):
        """Return boolean for outline."""
        return self.inputs.run_phonopy

    def is_nac(self):
        """Return boolean for outline."""
        if "nac" in self.inputs.calculator_inputs:
            return True
        if "is_nac" in self.inputs.settings.keys():
            self.logger.warning("Use inputs.settings['is_nac'] is deprecated.")
            return self.inputs.settings["is_nac"]
        return False

    def force_sets_exists(self):
        """Return boolean for outline."""
        return "force_sets" in self.inputs

    def nac_params_exists(self):
        """Return boolean for outline."""
        return "nac_params" in self.inputs

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

    def run_force_and_nac_calculations(self):
        """Run supercell force and NAC params calculations.

        Supercell force calculations and NAC params calculation
        are submitted in this method to make them run in parallel.

        """
        if "force_sets" in self.inputs:
            self.report("skip force calculation.")
            self.ctx.force_sets = self.inputs.force_sets
        else:
            self._run_force_calculations(self.ctx.supercells)
        if "nac_params" in self.inputs:
            self.report("skip nac params calculation.")
            self.ctx.nac_params = self.inputs.nac_params
        elif self.is_nac():
            self._run_nac_params_calculation()

    def _run_force_calculations(self, supercells, label_prefix="force_calc"):
        """Run supercell force calculations."""
        self.report("run force calculations")

        for key, supercell in supercells.items():
            num = key.split("_")[-1]
            label = f"{label_prefix}_{num}"
            builder = ForcesWorkChain.get_builder()
            builder.metadata.label = label
            builder.structure = supercell
            if "force" in self.inputs.calculator_inputs:
                calculator_inputs = self.inputs.calculator_inputs.force
            else:
                calculator_inputs = self.inputs.calculator_settings["forces"]
                self.logger.warning(
                    "Use calculator_inputs.force instead of "
                    "calculator_settings['forces']."
                )
            builder.calculator_inputs = calculator_inputs
            if "queue_name" in self.inputs:
                builder.queue_name = self.inputs.queue_name
            future = self.submit(builder)
            self.report("{} pk = {}".format(label, future.pk))
            self.to_context(**{label: future})

    def _run_nac_params_calculation(self):
        """Run nac params calculation."""
        self.report("run nac params calculation")

        builder = NacParamsWorkChain.get_builder()
        builder.metadata.label = "nac_params"
        builder.structure = self.ctx.primitive
        if "nac" in self.inputs.calculator_inputs:
            calculator_inputs = self.inputs.calculator_inputs.nac
        else:
            calculator_inputs = self.inputs.calculator_settings["nac"]
            self.logger.warning(
                "Use calculator_inputs.nac instead of calculator_settings['nac']."
            )

        builder.calculator_inputs = calculator_inputs
        if "queue_name" in self.inputs:
            builder.queue_name = self.inputs.queue_name
        future = self.submit(builder)
        self.report("nac_params: {}".format(future.pk))
        self.to_context(**{"nac_params_calc": future})

    def create_force_sets(self):
        """Build datasets from forces of supercells with displacments."""
        self.report("create force sets")

        forces_dict = collect_forces_and_energies(self.ctx, self.ctx.supercells)
        for key, val in get_force_sets(**forces_dict).items():
            self.ctx[key] = val
            self.out(key, self.ctx[key])

    def attach_nac_params(self):
        """Attach nac_params ArrayData to outputs."""
        self.report("create nac params")

        self.ctx.nac_params = self.ctx.nac_params_calc.outputs.nac_params
        self.out("nac_params", self.ctx.nac_params)

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
            builder.primitive = self.ctx.primitive
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
            "dos",
            "pdos",
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

        params = {}
        if "nac_params" in self.ctx:
            params["nac_params"] = self.ctx.nac_params
        result = get_phonon_properties(
            self.inputs.structure,
            self.ctx.phonon_setting_info,
            self.ctx.force_constants,
            self.inputs.symmetry_tolerance,
            **params,
        )
        self.out("thermal_properties", result["thermal_properties"])
        self.out("dos", result["dos"])
        self.out("band_structure", result["band_structure"])

        self.report("finish phonon")

    def finalize(self):
        """Show final message."""
        self.report("phonopy calculation has been done.")
