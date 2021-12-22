"""CalcJob to run phonopy at a remote host."""

from aiida.orm import BandsData, ArrayData, XyData, Dict, Str
from aiida_phonoxpy.calcs.base import BasePhonopyCalculation
from aiida.common import InputValidationError
from aiida_phonoxpy.common.file_generators import (
    get_FORCE_SETS_txt,
    get_phonopy_yaml_txt,
)


class PhonopyCalculation(BasePhonopyCalculation):
    """Phonopy calculation."""

    _OUTPUT_PROJECTED_DOS = "projected_dos.dat"
    _OUTPUT_TOTAL_DOS = "total_dos.dat"
    _OUTPUT_THERMAL_PROPERTIES = "thermal_properties.yaml"
    _OUTPUT_BAND_STRUCTURE = "band.yaml"
    _INOUT_FORCE_CONSTANTS = "force_constants.hdf5"
    _INPUT_CELL = "phonopy_cells.yaml"
    _INPUT_FORCE_SETS = "FORCE_SETS"

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        # parser_name has to be set to invoke parsing.
        spec.inputs["metadata"]["options"]["parser_name"].default = "phonopy"
        spec.inputs["metadata"]["options"]["output_filename"].default = "phonopy.yaml"

        spec.output(
            "force_constants",
            valid_type=ArrayData,
            required=False,
            help="Calculated force constants",
        )
        spec.output(
            "dos", valid_type=XyData, required=False, help="Calculated total DOS"
        )
        spec.output(
            "pdos", valid_type=XyData, required=False, help="Calculated projected DOS"
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

        self._create_phonopy_yaml(folder)
        self._create_FORCE_SETS(folder)
        mesh_opts, fc_opts = _get_phonopy_options(self.inputs.settings)

        if "displacements" in self.inputs:
            if "--alm" not in fc_opts:
                fc_opts.append("--alm")

        self._internal_retrieve_list = [
            self._INOUT_FORCE_CONSTANTS,
            self.inputs.metadata.options.output_filename,
        ]
        self._additional_cmd_params = [
            ["--writefc", "--writefc-format=hdf5"] + fc_opts,
            ["--readfc", "--readfc-format=hdf5"],
            ["--readfc", "--readfc-format=hdf5"],
        ]

        # First run with --writefc, and with --readfc for remaining runs
        if self.inputs.fc_only:
            self._calculation_cmd = [
                ["-c", self._INPUT_CELL],
            ]
        else:
            self._calculation_cmd = [
                ["-c", self._INPUT_CELL, "--pdos=auto"] + mesh_opts,
                ["-c", self._INPUT_CELL, "-t", "--dos"] + mesh_opts,
                [
                    "-c",
                    self._INPUT_CELL,
                    "--band=auto",
                    "--band-points=101",
                    "--band-const-interval",
                ],
            ]
            self._internal_retrieve_list += [
                self._OUTPUT_TOTAL_DOS,
                self._OUTPUT_PROJECTED_DOS,
                self._OUTPUT_THERMAL_PROPERTIES,
                self._OUTPUT_BAND_STRUCTURE,
            ]

    def _create_phonopy_yaml(self, folder):
        phpy_yaml_txt = get_phonopy_yaml_txt(
            self.inputs.structure,
            supercell_matrix=self.inputs.settings["supercell_matrix"],
        )
        with folder.open(self._INPUT_CELL, "w", encoding="utf8") as handle:
            handle.write(phpy_yaml_txt)

    def _create_FORCE_SETS(self, folder):
        if "force_sets" in self.inputs:
            force_sets = self.inputs.force_sets
        else:
            force_sets = None
        if "displacement_dataset" in self.inputs:
            dataset = self.inputs.displacement_dataset.get_dict()
        elif "displacements" in self.inputs:
            dataset = {
                "displacements": self.inputs.displacements.get_array("displacements")
            }
        else:
            dataset = None

        # can work both for type-I and type-II
        force_sets_txt = get_FORCE_SETS_txt(dataset, force_sets=force_sets)
        if force_sets_txt is None:
            msg = "Displacements or forces were not found."
            raise InputValidationError(msg)

        with folder.open(self._INPUT_FORCE_SETS, "w", encoding="utf8") as handle:
            handle.write(force_sets_txt)


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
