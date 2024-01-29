"""Parsers of phonopy output files."""

from aiida.common.exceptions import NotExistent
from aiida.engine import ExitCode
from aiida.orm import Str, SinglefileData
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

from aiida_phonoxpy.common.raw_parsers import (
    parse_band_structure,
    # parse_FORCE_CONSTANTS,
    parse_phonopy_yaml,
    parse_projected_dos,
    parse_thermal_properties,
    parse_total_dos,
)

PhonopyCalculation = CalculationFactory("phonoxpy.phonopy")


class PhonopyParser(Parser):
    """Parser the DATA files from phonopy."""

    def parse(self, **kwargs):
        """Parse retrieved files."""
        self.logger.info("parse retrieved files")

        # select the folder object
        # Check that the retrieved folder is there
        try:
            output_folder = self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        # check what is inside the folder
        list_of_files = output_folder.list_object_names()

        fc_filename = PhonopyCalculation._INOUT_FORCE_CONSTANTS
        # if fc_filename in list_of_files:
        #     with output_folder.open(fc_filename, "rb") as f:
        #         self.out("force_constants", parse_FORCE_CONSTANTS(f))
        if fc_filename in list_of_files:
            with output_folder.open(fc_filename, "rb") as handle:
                output_node = SinglefileData(file=handle, filename=fc_filename)
                self.out("force_constants", output_node)

        projected_dos_filename = PhonopyCalculation._OUTPUT_PROJECTED_DOS
        if projected_dos_filename in list_of_files:
            with output_folder.open(projected_dos_filename) as f:
                self.out("projected_dos", parse_projected_dos(f))

        total_dos_filename = PhonopyCalculation._OUTPUT_TOTAL_DOS
        if total_dos_filename in list_of_files:
            with output_folder.open(total_dos_filename) as f:
                self.out("total_dos", parse_total_dos(f))

        tp_filename = PhonopyCalculation._OUTPUT_THERMAL_PROPERTIES
        if tp_filename in list_of_files:
            with output_folder.open(tp_filename) as f:
                self.out("thermal_properties", parse_thermal_properties(f))

        band_filename = PhonopyCalculation._OUTPUT_BAND_STRUCTURE
        if band_filename in list_of_files:
            if "symmetry" in self.node.inputs.settings.attributes:
                sym_dataset = self.node.inputs.settings["symmetry"]
                label = "%s (%d)" % (
                    sym_dataset["international"],
                    sym_dataset["number"],
                )
            else:
                label = None
            with output_folder.open(band_filename) as f:
                self.out("band_structure", parse_band_structure(f, label=label))

        options = self.node.get_options()
        phpy_yaml_filename = options["output_filename"]
        if phpy_yaml_filename in list_of_files:
            with output_folder.open(phpy_yaml_filename) as f:
                yaml_dict = parse_phonopy_yaml(f)
                self.out("version", Str(yaml_dict["phonopy"]["version"]))

        self.logger.info("Parsing done.")
        return ExitCode(0)
