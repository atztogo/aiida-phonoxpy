"""Initialise a text database and profile for pytest.

Originally this was copied from aiida-quantumespresso.

"""
import os
import shutil
from collections.abc import Mapping

import numpy as np
import pytest

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]


@pytest.fixture(scope="session")
def filepath_tests():
    """Return the absolute filepath of the `tests` folder."""
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def filepath_fixtures(filepath_tests):
    """Return the absolute filepath to the directory containing the file `fixtures`."""
    return os.path.join(filepath_tests, "fixtures")


@pytest.fixture(scope="function")
def fixture_sandbox():
    """Return a `SandboxFolder`."""
    from aiida.common.folders import SandboxFolder

    with SandboxFolder() as folder:
        yield folder


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost `Computer`."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture
def fixture_code(fixture_localhost):
    """Return a `Code` instance.

    This is configured to run calculations of given entry point on localhost
    `Computer`.

    """

    def _fixture_code(entry_point_name):
        from aiida.common import exceptions
        from aiida.orm import Code

        label = f"test.{entry_point_name}"

        try:
            return Code.objects.get(label=label)
        except exceptions.NotExistent:
            return Code(
                label=label,
                input_plugin_name=entry_point_name,
                remote_computer_exec=[fixture_localhost, "/bin/true"],
            )

    return _fixture_code


@pytest.fixture
def generate_structure():
    """Return a `StructureData` of conventional unit cell of NaCl."""

    def _generate_structure(structure_id="NaCl"):
        from aiida.orm import StructureData

        if structure_id == "NaCl":
            a = 5.6903014761756712
            structure = StructureData(
                cell=[
                    [a, 0.0, 0.0],
                    [0.0, a, 0.0],
                    [0.0, 0.0, a],
                ]
            )
            lattice = structure.cell
            pos_Na = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
            for pos in pos_Na:
                structure.append_atom(
                    position=np.dot(pos, lattice), symbols="Na", name="Na"
                )
            pos_Cl = [[0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
            for pos in pos_Cl:
                structure.append_atom(
                    position=np.dot(pos, lattice), symbols="Cl", name="Cl"
                )
        else:
            raise KeyError(f'Unknown structure_id="{structure_id}"')
        return structure

    return _generate_structure


@pytest.fixture
def generate_nac_params():
    """Return a `ArrayData` of NAC params of NaCl."""

    def _generate_nac_params(structure_id="NaCl"):
        from aiida.orm import ArrayData

        if structure_id == "NaCl":
            nac_params = ArrayData()
            _I = np.eye(3)
            nac_params.set_array(
                "born_charges", np.array([1.10268732 * _I, -1.10268732 * _I])
            )
            nac_params.set_array("epsilon", np.array(2.48006321 * _I))
        else:
            raise KeyError(f'Unknown structure_id="{structure_id}"')
        return nac_params

    return _generate_nac_params


@pytest.fixture
def generate_phonopy_settings():
    """Return a `Dict` of phonopy settings."""

    def _generate_phonopy_settings(supercell_matrix=None, number_of_snapshots=None):
        from aiida.orm import Dict

        if supercell_matrix is None:
            _supercell_matrix = [1, 1, 1]
        else:
            _supercell_matrix = supercell_matrix
        settings = {
            "mesh": 50.0,
            "supercell_matrix": _supercell_matrix,
            "distance": 0.03,
        }
        if number_of_snapshots is not None:
            settings["number_of_snapshots"] = number_of_snapshots
        return Dict(dict=settings)

    return _generate_phonopy_settings


@pytest.fixture
def generate_kpoints_mesh():
    """Return a `KpointsData` node."""

    def _generate_kpoints_mesh(npoints):
        """Return a `KpointsData` with a mesh of npoints in each direction."""
        from aiida.orm import KpointsData

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([npoints] * 3)

        return kpoints

    return _generate_kpoints_mesh


@pytest.fixture(scope="session")
def generate_parser():
    """Fixture to load a parser class for testing parsers."""

    def _generate_parser(entry_point_name):
        """Fixture to load a parser class for testing parsers.

        :param entry_point_name: entry point name of the parser class
        :return: the `Parser` sub class
        """
        from aiida.plugins import ParserFactory

        return ParserFactory(entry_point_name)

    return _generate_parser


@pytest.fixture
def generate_remote_data():
    """Return a `RemoteData` node."""

    def _generate_remote_data(computer, remote_path, entry_point_name=None):
        """Return a `KpointsData` with a mesh of npoints in each direction."""
        from aiida.common.links import LinkType
        from aiida.orm import CalcJobNode, RemoteData
        from aiida.plugins.entry_point import format_entry_point_string

        entry_point = format_entry_point_string("aiida.calculations", entry_point_name)

        remote = RemoteData(remote_path=remote_path)
        remote.computer = computer

        if entry_point_name is not None:
            creator = CalcJobNode(computer=computer, process_type=entry_point)
            creator.set_option(
                "resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1}
            )
            remote.add_incoming(
                creator, link_type=LinkType.CREATE, link_label="remote_folder"
            )
            creator.store()

        return remote

    return _generate_remote_data


@pytest.fixture
def generate_workchain():
    """Generate an instance of a `WorkChain`."""

    def _generate_workchain(entry_point, inputs):
        """Generate an instance of a `WorkChain` with the given entry point and inputs.

        Parameters
        ----------
        entry_point : str
            Entry point name of the work chain subclass.
        inputs : dict
            Inputs to be passed to process construction.

        Returns
        -------
        WorkChain

        """
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import WorkflowFactory

        process_class = WorkflowFactory(entry_point)
        runner = get_manager().get_runner()
        process = instantiate_process(runner, process_class, **inputs)

        return process

    return _generate_workchain


@pytest.fixture
def generate_calc_job_node(fixture_localhost):
    """Fixture to generate a mock `CalcJobNode` for testing parsers."""

    def flatten_inputs(inputs, prefix=""):
        """Flatten inputs recursively.

        This works like :meth:`aiida.engine.processes.process::Process._flatten_inputs`.

        Parameters
        ----------
        inputs : dict
            Any optional nodes to add as input links to the corrent CalcJobNode.

        Returns
        -------
        list
            Flattened inputs.

        """
        flat_inputs = []
        for key, value in inputs.items():
            if isinstance(value, Mapping):
                flat_inputs.extend(flatten_inputs(value, prefix=prefix + key + "__"))
            else:
                flat_inputs.append((prefix + key, value))
        return flat_inputs

    def _generate_calc_job_node(
        entry_point_name="phonoxpy.phonopy",
        computer=None,
        test_name=None,
        inputs=None,
        attributes=None,
        retrieve_temporary=None,
    ):
        """Fixture to generate a mock `CalcJobNode` for testing parsers.

        Parameters
        ----------
        entry_point_name : str
            Entry point name of the calculation class
        computer : Computer
        test_name : str
            Relative path of directory with test output files in the
            `fixtures/{entry_point_name}` folder.
        inputs : dict
            Any optional nodes to add as input links to the corrent CalcJobNode.
        attributes : Any optional attributes to set on the node.
        retrieve_temporary : tuple, optional
            An absolute filepath of a temporary directory and a list of filenames that
            should be written to this directory, which will serve as the
            `retrieved_temporary_folder`. For now this only works with top-level files
            and does not support files nested in directories.

        returns
        -------
        CalcJobNode :
            Instance with an attached `FolderData` as the `retrieved` node.

        """
        from aiida.common import LinkType
        from aiida.orm import CalcJobNode, Dict, FolderData, RemoteData
        from aiida.plugins.entry_point import format_entry_point_string

        if computer is None:
            computer = fixture_localhost

        filepath_folder = None

        if test_name is not None:
            basepath = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(entry_point_name[len("phonoxpy.") :], test_name)
            filepath_folder = os.path.join(basepath, "parsers", "fixtures", filename)
            filepath_input = os.path.join(filepath_folder, "phonopy_params.yaml")

        entry_point = format_entry_point_string("aiida.calculations", entry_point_name)

        node = CalcJobNode(computer=computer, process_type=entry_point)
        # node.set_attribute("input_filename", "aiida.in")
        node.set_attribute("output_filename", "phonopy.yaml")
        # node.set_attribute("error_filename", "aiida.err")
        node.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
        node.set_option("max_wallclock_seconds", 1800)

        if attributes:
            node.set_attribute_many(attributes)

        if filepath_folder:
            from phonopy.interface.calculator import read_crystal_structure

            from aiida_phonoxpy.common.utils import phonopy_atoms_to_structure

            phonopy_cell, _ = read_crystal_structure(
                filepath_input, interface_mode="phonopy_yaml"
            )
            inputs["structure"] = phonopy_atoms_to_structure(phonopy_cell)
            supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
            inputs["settings"] = Dict(dict={"supercell_matrix": supercell_matrix})

        if inputs:
            metadata = inputs.pop("metadata", {})
            options = metadata.get("options", {})

            for name, option in options.items():
                node.set_option(name, option)

            for link_label, input_node in flatten_inputs(inputs):
                input_node.store()
                node.add_incoming(
                    input_node, link_type=LinkType.INPUT_CALC, link_label=link_label
                )

        node.store()

        if retrieve_temporary:
            dirpath, filenames = retrieve_temporary
            for filename in filenames:
                try:
                    shutil.copy(
                        os.path.join(filepath_folder, filename),
                        os.path.join(dirpath, filename),
                    )
                # To test the absence of files in the retrieve_temporary folder
                except FileNotFoundError:
                    pass

        if filepath_folder:
            retrieved = FolderData()
            retrieved.put_object_from_tree(filepath_folder)

            # Remove files that are supposed to be only present in the retrieved
            # temporary folder
            if retrieve_temporary:
                for filename in filenames:
                    try:
                        retrieved.delete_object(filename)
                    # To test the absence of files in the retrieve_temporary folder
                    except OSError:
                        pass

            retrieved.add_incoming(
                node, link_type=LinkType.CREATE, link_label="retrieved"
            )
            retrieved.store()

            remote_folder = RemoteData(computer=computer, remote_path="/tmp")
            remote_folder.add_incoming(
                node, link_type=LinkType.CREATE, link_label="remote_folder"
            )
            remote_folder.store()

        return node

    return _generate_calc_job_node
