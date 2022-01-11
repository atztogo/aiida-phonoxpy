"""Base class of PhonopyCalculation and Phono3pyCalculation."""

from aiida.common import CalcInfo, CodeInfo
from aiida.engine import CalcJob
from aiida.orm import ArrayData, Dict, Float, StructureData


class BasePhonopyCalculation(CalcJob):
    """A basic plugin for calculating force constants using Phonopy.

    Requirement: the node should be able to import phonopy if NAC is used

    """

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)
        spec.input("metadata.options.withmpi", valid_type=bool, default=False)
        spec.input(
            "structure",
            valid_type=StructureData,
            required=True,
            help="Unit cell structure.",
        )
        spec.input(
            "settings", valid_type=Dict, required=True, help="Phono3py parameters."
        )
        spec.input("symmetry_tolerance", valid_type=Float, default=lambda: Float(1e-5))
        spec.input(
            "force_sets",
            valid_type=ArrayData,
            required=False,
            help="Sets of forces in supercells",
        )
        spec.input(
            "nac_params", valid_type=ArrayData, required=False, help="NAC parameters."
        )
        spec.input(
            "displacement_dataset",
            valid_type=Dict,
            required=False,
            help="Type-I displacement dataset.",
        )
        spec.input(
            "displacements",
            valid_type=ArrayData,
            required=False,
            help="Displacements of all atoms corresponding to force_sets.",
        )

    def prepare_for_submission(self, folder):
        """Prepare calcinfo."""
        self.logger.info("prepare_for_submission")

        # These three lists are updated in self._create_additional_files(folder)
        self._internal_retrieve_list = []
        self._additional_cmd_params = []
        self._calculation_cmd = []

        self._create_additional_files(folder)
        self._set_commands_and_retrieve_list()

        if "nac_params" in self.inputs:
            for params in self._additional_cmd_params:
                params.append("--nac")

        # ============================ calcinfo ===============================

        local_copy_list = []
        remote_copy_list = []

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.retrieve_list = self._internal_retrieve_list

        calcinfo.codes_info = []
        for i, (default_params, additional_params) in enumerate(
            zip(self._calculation_cmd, self._additional_cmd_params)
        ):
            codeinfo = CodeInfo()
            cmdline_params = default_params + additional_params
            codeinfo.cmdline_params = cmdline_params
            codeinfo.code_uuid = self.inputs.code.uuid
            codeinfo.withmpi = False
            calcinfo.codes_info.append(codeinfo)

        return calcinfo

    def _create_additional_files(self, folder):
        raise NotImplementedError()

    def _set_commands(self):
        raise NotImplementedError()

    def _get_nac_params(self, nac_data):
        born_charges = nac_data.get_array("born_charges")
        epsilon = nac_data.get_array("epsilon")
        return {"born": born_charges, "dielectric": epsilon}

    def _set_dataset(self, ph, prefix=""):
        """Prefix can be `` or `phonon_`."""
        key_dataset = prefix + "displacement_dataset"
        key_disps = prefix + "displacements"
        key_forces = prefix + "forces"
        key_force_sets = prefix + "force_sets"
        if key_dataset in self.inputs:
            setattr(ph, prefix + "dataset", self.inputs[key_dataset].get_dict())
        elif key_disps in self.inputs:
            setattr(
                ph,
                prefix + "dataset",
                {"displacements": self.inputs[key_disps].get_array("displacements")},
            )
        if key_force_sets in self.inputs:
            setattr(ph, key_forces, self.inputs[key_force_sets].get_array("force_sets"))
