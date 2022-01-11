"""Parsers of phonopy output files."""
from aiida.common.exceptions import NotExistent
from aiida.engine import ExitCode
from aiida.orm import SinglefileData, Str
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

from aiida_phonoxpy.common.raw_parsers import parse_phonopy_yaml

Phono3pyCalculation = CalculationFactory("phonoxpy.phono3py")


class Phono3pyParser(Parser):
    """Parser the DATA files from phonopy."""

    def parse(self, **kwargs):
        """Parse retrieved files."""
        self.logger.info("parse retrieved files")

        try:
            output_folder = self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        list_of_filenames = output_folder.list_object_names()
        options = self.node.get_options()

        kappa_filename = None
        if "fc2" in self.node.inputs and "fc3" in self.node.inputs:
            for filename in list_of_filenames:
                if "kappa" in filename:
                    kappa_filename = filename
                    filenames_parsed = (options["output_filename"], kappa_filename)
                    break
            if not kappa_filename:
                filenames_parsed = (options["output_filename"],)
        else:
            filenames_parsed = (options["output_filename"], "fc2.hdf5", "fc3.hdf5")

        for filename in filenames_parsed:
            if filename not in list_of_filenames:
                return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        with output_folder.open(options["output_filename"]) as f:
            yaml_dict = parse_phonopy_yaml(f)
            self.out("version", Str(yaml_dict["phono3py"]["version"]))

        for filename in ("fc2.hdf5", "fc3.hdf5"):
            if filename in filename in filenames_parsed:
                with output_folder.open(filename, "rb") as handle:
                    output_node = SinglefileData(file=handle)
                    self.out(filename.replace(".hdf5", ""), output_node)

        if kappa_filename:
            with output_folder.open(filename, "rb") as handle:
                output_node = SinglefileData(file=handle)
                self.out("ltc", output_node)

        self.logger.info("Parsing done.")
        return ExitCode(0)
