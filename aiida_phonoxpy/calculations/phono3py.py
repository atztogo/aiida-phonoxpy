"""CalcJob to run phonopy at a remote host."""
import lzma

from aiida.orm import ArrayData, Dict, Str

from aiida_phonoxpy.calculations.base import BasePhonopyCalculation
from aiida_phonoxpy.utils.utils import get_phono3py_instance


class Phono3pyCalculation(BasePhonopyCalculation):
    """Phonopy calculation."""

    _INOUT_FC2 = "fc2.hdf5"
    _INOUT_FC3 = "fc3.hdf5"
    _INPUT_PARAMS = "phono3py_params.yaml.xz"

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        # parser_name has to be set to invoke parsing.
        spec.input("metadata.options.parser_name", default="phonoxpy.phono3py")
        spec.input("metadata.options.output_filename", default="phono3py.yaml")

        spec.output(
            "fc3",
            valid_type=ArrayData,
            required=False,
            help="Calculated third order force constants",
        )
        spec.output(
            "fc2",
            valid_type=ArrayData,
            required=False,
            help="Calculated second order force constants",
        )
        spec.output("version", valid_type=Str, required=False, help="Version number")

    def prepare_for_submission(self, folder):
        """Prepare calcinfo."""
        return super().prepare_for_submission(folder)

    def _create_additional_files(self, folder):
        from phono3py.interface.phono3py_yaml import Phono3pyYaml

        self.logger.info("create_additional_files")

        ph3 = self._get_phono3py_instance()
        ph3py_yaml = Phono3pyYaml(settings={"force_sets": True})
        ph3py_yaml.set_phonon_info(ph3)
        with folder.open(self._INPUT_PARAMS, "wb") as handle:
            handle.write(lzma.compress(str(ph3py_yaml).encode()))

    def _set_commands_and_retrieve_list(self):
        mesh_opts, fc_opts = _get_phonopy_options(self.inputs.settings)
        if "displacements" in self.inputs:
            if "--alm" not in fc_opts:
                fc_opts.append("--alm")

        self._internal_retrieve_list = [
            self.inputs.metadata.options.output_filename,
        ]
        self._additional_cmd_params = [
            fc_opts,
        ]

        # First run with --writefc, and with --readfc for remaining runs
        if self.inputs.fc_only:
            self._calculation_cmd = [
                ["-c", self._INPUT_PARAMS],
            ]
        else:
            self._calculation_cmd = [
                ["-c", self._INPUT_PARAMS],
            ]

    def _get_phono3py_instance(self):
        nac_params = None
        if not self.inputs.fc_only and "nac_params" in self.inputs:
            nac_params = self.inputs.nac_params
        ph3 = get_phono3py_instance(
            self.inputs.structure,
            self.inputs.settings.get_dict(),
            nac_params=nac_params,
        )
        self._set_dataset(ph3)
        self._set_dataset(ph3, prefix="phonon_")
        return ph3


def _get_phonopy_options(settings: Dict):
    """Return phonopy command options as strings."""
    mesh_opts = []
    if "mesh" in settings.keys():
        mesh = settings["mesh"]
        try:
            length = float(mesh)
            mesh_opts.append("--mesh=%f" % length)
        except TypeError:
            mesh_opts.append('--mesh="%d %d %d"' % tuple(mesh))
        mesh_opts.append("--nowritemesh")

    fc_opts = []
    if "fc_calculator" in settings.keys():
        if settings["fc_calculator"].lower().strip() == "alm":
            fc_opts.append("--alm")
    return mesh_opts, fc_opts
