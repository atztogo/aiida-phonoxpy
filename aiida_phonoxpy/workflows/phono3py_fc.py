"""WorkChan to calculate force constants by phono3py."""

from aiida.engine import WorkChain
from aiida.orm import ArrayData, Code, Dict, Float, StructureData
from aiida.orm.nodes.data.singlefile import SinglefileData

from aiida_phonoxpy.utils.utils import setup_phono3py_fc_calculation
from aiida_phonoxpy.workflows.mixin import RunPhono3pyMixIn


class Phono3pyFCWorkChain(WorkChain, RunPhono3pyMixIn):
    """Force constants workchain.

    This workchain generates fc2 and fc3.

    """

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)
        spec.input("structure", valid_type=StructureData, required=True)
        spec.input("settings", valid_type=Dict, required=True)
        spec.input("symmetry_tolerance", valid_type=Float, default=lambda: Float(1e-5))
        spec.input("code", valid_type=Code, required=False)
        spec.input("displacement_dataset", valid_type=Dict, required=False)
        spec.input("displacements", valid_type=ArrayData, required=False)
        spec.input("force_sets", valid_type=ArrayData, required=False)
        spec.input("phonon_force_sets", valid_type=ArrayData, required=False)
        spec.input("phonon_displacement_dataset", valid_type=Dict, required=False)
        spec.input("phonon_displacements", valid_type=ArrayData, required=False)
        spec.input("phono3py.metadata", valid_type=dict, required=False, non_db=True)

        spec.outline(
            cls.initialize,
            cls.run_phono3py_fc,
            cls.collect_fc,
            cls.finalize,
        )
        spec.output("fc3", valid_type=SinglefileData, required=False)
        spec.output("fc2", valid_type=SinglefileData, required=False)
        spec.output("phonon_setting_info", valid_type=Dict, required=False)

    def initialize(self):
        """Set default settings and create supercells and primitive cell."""
        self.report("initialize")

        for key in (
            "displacement_dataset",
            "displacements",
            "force_sets",
            "phonon_displacement_dataset",
            "phonon_displacements",
            "phonon_force_sets",
        ):
            if key in self.inputs:
                self.ctx[key] = self.inputs[key]
        return_vals = setup_phono3py_fc_calculation(
            self.inputs.settings,
            self.inputs.symmetry_tolerance,
        )

        key = "phonon_setting_info"
        if key in return_vals:
            self.ctx[key] = return_vals[key]
            self.out(key, self.ctx[key])

    def run_phono3py_fc(self):
        """Run phonopy to calculate fc3 and fc2."""
        self.report("run fc3 and fc2 calculations.")
        self._run_phono3py(calc_type="fc")

    def collect_fc(self):
        """Collect fc2 and fc3."""
        for key in ("fc2", "fc3"):
            if key in self.ctx.fc_calc.outputs:
                self.ctx[key] = self.ctx.fc_calc.outputs[key]
                self.out(key, self.ctx[key])

    def finalize(self):
        """Show final message."""
        self.report("phono3py force constants calculation has been done.")
