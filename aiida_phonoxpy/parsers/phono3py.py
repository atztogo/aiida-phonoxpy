"""Parsers of phonopy output files."""
from aiida.common.exceptions import NotExistent
from aiida.engine import ExitCode
from aiida.orm import Str
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

from aiida_phonoxpy.common.raw_parsers import parse_phonopy_yaml

Phono3pyCalculation = CalculationFactory("phonoxpy.phono3py")


class Phono3pyParser(Parser):
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

        options = self.node.get_options()
        ph3py_yaml_filename = options["output_filename"]
        if ph3py_yaml_filename in list_of_files:
            with output_folder.open(ph3py_yaml_filename) as f:
                yaml_dict = parse_phonopy_yaml(f)
                self.out("version", Str(yaml_dict["phono3py"]["version"]))

        self.logger.info("Parsing done.")
        return ExitCode(0)
