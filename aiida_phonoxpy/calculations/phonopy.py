"""CalcJob to run phonopy at a remote host."""
import lzma

from aiida.orm import ArrayData, BandsData, Dict, Str, XyData
from phonopy.interface.phonopy_yaml import PhonopyYaml

from aiida_phonoxpy.calculations.base import BasePhonopyCalculation
from aiida_phonoxpy.utils.utils import get_phonopy_instance


class PhonopyCalculation(BasePhonopyCalculation):
    """Phonopy calculation."""

    _OUTPUT_PROJECTED_DOS = "projected_dos.dat"
    _OUTPUT_TOTAL_DOS = "total_dos.dat"
    _OUTPUT_THERMAL_PROPERTIES = "thermal_properties.yaml"
    _OUTPUT_BAND_STRUCTURE = "band.yaml"
    _INOUT_FORCE_CONSTANTS = "force_constants.hdf5"
    _INPUT_PARAMS = "phonopy_params.yaml.xz"

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        # parser_name has to be set to invoke parsing.
        spec.input("metadata.options.parser_name", default="phonoxpy.phonopy")
        spec.input("metadata.options.output_filename", default="phonopy.yaml")

        spec.output(
            "force_constants",
            valid_type=ArrayData,
            required=False,
            help="Calculated force constants",
        )
        spec.output(
            "total_dos", valid_type=XyData, required=False, help="Calculated total DOS"
        )
        spec.output(
            "projected_dos",
            valid_type=XyData,
            required=False,
            help="Calculated projected DOS",
        )
        spec.output(
            "thermal_properties",
            valid_type=XyData,
            required=False,
            help="Calculated thermal properties",
        )
        spec.output(
            "band_structure",
            valid_type=BandsData,
            required=False,
            help="Calculated phonon band structure",
        )
        spec.output("version", valid_type=Str, required=False, help="Version number")

    def prepare_for_submission(self, folder):
        """Prepare calcinfo."""
        return super().prepare_for_submission(folder)

    def _create_additional_files(self, folder):
        self.logger.info("create_additional_files")

        ph = self._get_phonopy_instance()
        phpy_yaml = PhonopyYaml()
        phpy_yaml.set_phonon_info(ph)

        with folder.open(self._INPUT_PARAMS, "wb") as handle:
            handle.write(lzma.compress(str(phpy_yaml).encode()))

    def _set_commands_and_retrieve_list(self):
        mesh_opts, fc_opts = _get_phonopy_options(self.inputs.settings)
        if "displacements" in self.inputs:
            if "--alm" not in fc_opts:
                fc_opts.append("--alm")

        self._internal_retrieve_list = [
            self._INOUT_FORCE_CONSTANTS,
            self.inputs.metadata.options.output_filename,
        ]

        self._additional_cmd_params = [["--writefc", "--writefc-format=hdf5"] + fc_opts]

        if mesh_opts:
            self._calculation_cmd = [
                ["-c", self._INPUT_PARAMS, "--pdos=auto"] + mesh_opts,
                ["-c", self._INPUT_PARAMS, "-t"] + mesh_opts,
                [
                    "-c",
                    self._INPUT_PARAMS,
                    "--band=auto",
                    "--band-points=101",
                    "--band-const-interval",
                ],
            ]
            self._internal_retrieve_list += [
                self._OUTPUT_PROJECTED_DOS,
                self._OUTPUT_THERMAL_PROPERTIES,
                self._OUTPUT_BAND_STRUCTURE,
            ]
            self._additional_cmd_params += [
                ["--readfc", "--readfc-format=hdf5"],
                ["--readfc", "--readfc-format=hdf5"],
            ]
        else:
            self._calculation_cmd = [
                ["-c", self._INPUT_PARAMS],
            ]

    def _get_phonopy_instance(self):
        nac_params = None
        if "nac_params" in self.inputs:
            nac_params = self.inputs.nac_params
        ph = get_phonopy_instance(
            self.inputs.structure,
            self.inputs.settings.get_dict(),
            nac_params=nac_params,
        )
        self._set_dataset(ph)
        return ph


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
