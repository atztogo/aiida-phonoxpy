"""WorkChan to calculate lattice thermal conductivity by phono3py."""

from aiida.engine import WorkChain
from aiida.orm import ArrayData, Code, Dict, Float, StructureData
from aiida.orm.nodes.data.singlefile import SinglefileData

from aiida_phonoxpy.calculations.phono3py import Phono3pyCalculation
from aiida_phonoxpy.utils.utils import setup_phono3py_ltc_calculation
from aiida_phonoxpy.workflows.mixin import RunPhono3pyMixIn


class Phono3pyLTCWorkChain(WorkChain, RunPhono3pyMixIn):
    """Lattice thermal conductivity workchain.

    This workchain calculates lattice thermal conductivity.

    """

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)
        spec.input("structure", valid_type=StructureData, required=True)
        spec.input("settings", valid_type=Dict, required=True)
        spec.expose_inputs(
            Phono3pyCalculation, namespace="phono3py", include=("metadata",)
        )
        spec.input(
            "phono3py.metadata.options.resources", valid_type=dict, required=False
        )
        spec.input("symmetry_tolerance", valid_type=Float, default=lambda: Float(1e-5))
        spec.input("code", valid_type=Code, required=False)
        spec.input("fc3", valid_type=SinglefileData, required=False)
        spec.input("fc2", valid_type=SinglefileData, required=False)
        spec.input("nac_params", valid_type=ArrayData, required=False)

        spec.outline(
            cls.initialize,
            cls.run_phono3py_ltc,
            cls.collect_ltc,
            cls.finalize,
        )
        spec.output("phonon_setting_info", valid_type=Dict, required=False)
        spec.output("ltc", valid_type=SinglefileData, required=False)

    def initialize(self):
        """Set default settings and create supercells and primitive cell.

        return_vals['phonon_setting_info'] : Dict

        """
        self.report("initialize")

        for key in ("fc2", "fc3", "nac_params"):
            if key in self.inputs:
                self.ctx[key] = self.inputs[key]

        return_vals = setup_phono3py_ltc_calculation(
            self.inputs.settings,
            self.inputs.structure,
            self.inputs.symmetry_tolerance,
        )

        key = "phonon_setting_info"
        if key in return_vals:
            self.ctx[key] = return_vals[key]
            self.out(key, self.ctx[key])

    def run_phono3py_ltc(self):
        """Run phonopy to calculate LTC."""
        self.report("run LTC calculations.")
        self._run_phono3py(calc_type="ltc")

    def collect_ltc(self):
        """Collect LTC."""
        key = "ltc"
        if key in self.ctx.ltc_calc.outputs:
            self.ctx[key] = self.ctx.ltc_calc.outputs[key]
            self.out(key, self.ctx[key])

    def finalize(self):
        """Show final message."""
        self.report("phono3py LTC calculation has been done.")
