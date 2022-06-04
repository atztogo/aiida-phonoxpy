"""CalcJob to run phonopy at a remote host."""
import lzma
import logging
from aiida.orm import ArrayData, Dict, SinglefileData, Str

from aiida_phonoxpy.calculations.base import BasePhonopyCalculation
from aiida_phonoxpy.utils.utils import get_phono3py_instance


class Phono3pyCalculation(BasePhonopyCalculation):
    """Phonopy calculation."""

    _INOUT_FC2 = "fc2.hdf5"
    _INOUT_FC3 = "fc3.hdf5"
    _INPUT_PARAMS = "phono3py_params.yaml.xz"
    _OUTPUT_LTC = "kappa-*.hdf5"

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        # parser_name has to be set to invoke parsing.
        spec.input("metadata.options.parser_name", default="phonoxpy.phono3py")
        spec.input("metadata.options.output_filename", default="phono3py.yaml")
        spec.input(
            "phonon_force_sets",
            valid_type=ArrayData,
            required=False,
            help="Sets of forces in supercells for fc2",
        )
        spec.input(
            "phonon_displacement_dataset",
            valid_type=Dict,
            required=False,
            help="Type-I displacement dataset for fc2.",
        )
        spec.input(
            "phonon_displacements",
            valid_type=ArrayData,
            required=False,
            help="Displacements of all atoms corresponding to force_sets for fc2.",
        )
        spec.input(
            "fc3",
            valid_type=SinglefileData,
            required=False,
            help="Third order force constants (fc3.hdf5)",
        )
        spec.input(
            "fc2",
            valid_type=SinglefileData,
            required=False,
            help="Second order force constants (fc2.hdf5)",
        )

        spec.output(
            "fc3",
            valid_type=SinglefileData,
            required=False,
            help="Third order force constants (fc3.hdf5).",
        )
        spec.output(
            "fc2",
            valid_type=SinglefileData,
            required=False,
            help="Second order force constants (fc2.hdf5)",
        )
        spec.output(
            "ltc",
            valid_type=SinglefileData,
            required=False,
            help="Lattice thermal conductivity (kappa-xxx.hdf5)",
        )
        spec.output("version", valid_type=Str, required=False, help="Version number")

    def prepare_for_submission(self, folder):
        """Prepare calcinfo."""
        calcinfo = super().prepare_for_submission(folder)
        for key in ("fc3", "fc2"):
            if key in self.inputs:
                fc_file = self.inputs[key]
                if isinstance(fc_file, SinglefileData):
                    calcinfo.local_copy_list.append(
                        (fc_file.uuid, fc_file.filename, f"{key}.hdf5")
                    )
        return calcinfo

    def _create_additional_files(self, folder):
        from phono3py.interface.phono3py_yaml import Phono3pyYaml

        self.logger.info("create_additional_files")

        ph3 = self._get_phono3py_instance()
        ph3py_yaml = Phono3pyYaml(settings={"force_sets": True})
        ph3py_yaml.set_phonon_info(ph3)
        with folder.open(self._INPUT_PARAMS, "wb") as handle:
            handle.write(lzma.compress(str(ph3py_yaml).encode()))

    def _set_commands_and_retrieve_list(self):
        self._internal_retrieve_list = [
            self.inputs.metadata.options.output_filename,
        ]

        opts_dict = _get_phono3py_options(self.inputs.settings, self.logger)
        fc_opts = opts_dict["fc"]
        mesh_opts = opts_dict["mesh"]
        isotope_opts = opts_dict["isotope"]
        temperature_opts = opts_dict["temperature"]
        conductivity_opts = opts_dict["conductivity"]

        # Assume LTC calculation
        if "fc2" in self.inputs and "fc3" in self.inputs:
            comm_opts = (
                ["--fc2", "--fc3"]
                + conductivity_opts
                + mesh_opts
                + temperature_opts
                + isotope_opts
            )
            self._internal_retrieve_list.append(self._OUTPUT_LTC)
        else:  # Assume force constants calculation
            if "displacements" in self.inputs:
                if "--alm" not in fc_opts:
                    fc_opts.append("--alm")
            if "--alm" not in fc_opts:
                fc_opts.append("--sym-fc")
            fc_opts.append("--compact-fc")
            for key in ("fc2", "fc3"):
                if key in self.inputs:
                    fc_opts.append(f"--{key}")
                elif key == "fc2":
                    self._internal_retrieve_list.append(self._INOUT_FC2)
                elif key == "fc3":
                    self._internal_retrieve_list.append(self._INOUT_FC3)
            comm_opts = fc_opts

        self._additional_cmd_params = [comm_opts]
        self._calculation_cmd = [["-c", self._INPUT_PARAMS]]

    def _get_phono3py_instance(self):
        kwargs = {}
        if "nac_params" in self.inputs:
            kwargs.update({"nac_params": self.inputs.nac_params})
        ph3 = get_phono3py_instance(
            self.inputs.structure, self.inputs.settings.get_dict(), **kwargs
        )
        self._set_dataset(ph3)
        self._set_dataset(ph3, prefix="phonon_")
        return ph3


def _get_phono3py_options(settings: Dict, logger: logging.Logger) -> dict:
    """Return phonopy command options as strings."""
    mesh_opts = []
    fc_opts = []
    isotope_opts = []
    temperature_opts = ["--ts"]
    conductivity_opts = []

    if "mesh" in settings.keys():
        mesh = settings["mesh"]
        try:
            length = float(mesh)
            mesh_opts += ["--mesh", f"{length}"]
        except TypeError:
            mesh_opts += ["--mesh", f"{mesh[0]}", f"{mesh[1]}", f"{mesh[2]}"]
    else:
        logger.info("mesh setting not found. Set mesh=30.")
        mesh_opts += ["--mesh", "30"]

    if "grg" in settings.keys():
        if settings["grg"]:
            mesh_opts.append("--grg")

    if "fc_calculator" in settings.keys():
        if settings["fc_calculator"].lower().strip() == "alm":
            fc_opts.append("-v")
            fc_opts.append("--alm")
            if "fc_calculator_options" in settings.keys():
                fc_calc_opts = settings["fc_calculator_options"]
                fc_opts += ["--fc-calculator-options", f"{fc_calc_opts}"]

    if "ts" in settings.keys():
        temperature_opts += [str(t) for t in settings["ts"]]
    else:
        temperature_opts += ["300"]

    if "br" in settings.keys() and settings["br"]:
        conductivity_opts += ["--br"]
    elif "lbte" in settings.keys() and settings["lbte"]:
        conductivity_opts += ["--lbte"]
    else:
        conductivity_opts += ["--br"]

    if "isotope" in settings.keys():
        if settings["isotope"]:
            isotope_opts.append("--isotope")

    return {
        "mesh": mesh_opts,
        "fc": fc_opts,
        "isotope": isotope_opts,
        "temperature": temperature_opts,
        "conductivity": conductivity_opts,
    }
