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
def generate_displacements():
    """Return a `ArrayData` of displacements of NaCl."""

    def _generate_displacements(structure_id="NaCl"):
        from aiida.orm import ArrayData

        if structure_id == "NaCl":
            displacements = ArrayData()
            displacements.set_array(
                "displacements",
                np.array(
                    [
                        [
                            [-0.02273746, -0.01931934, -0.00312583],
                            [-0.0228422, 0.0193765, 0.00166885],
                            [0.0224281, 0.01592108, 0.01197912],
                            [-0.00064646, 0.00217048, -0.0299144],
                            [0.01862151, -0.01785433, -0.01531217],
                            [0.00067696, 0.00025026, 0.02999132],
                            [-0.01435842, -0.02476933, 0.00896193],
                            [0.00253315, -0.00924756, 0.0284265],
                        ],
                        [
                            [-0.01740268, -0.02413313, 0.00383909],
                            [0.0170104, -0.00024372, -0.02471006],
                            [0.01743549, -0.01667561, -0.01783052],
                            [-0.0135311, 0.02481529, 0.01005539],
                            [0.01114619, -0.00962554, -0.0261364],
                            [0.0214863, 0.02083848, 0.00202403],
                            [-0.0225386, -0.01897413, 0.0056563],
                            [-0.02252869, -0.01415487, 0.01385993],
                        ],
                        [
                            [-0.01523419, 0.0214319, -0.01444275],
                            [-0.00511404, -0.02951603, -0.00162813],
                            [0.01960127, 0.01683884, 0.01523954],
                            [0.01784544, -0.00675864, -0.02314867],
                            [-0.00760845, -0.01536795, 0.0246158],
                            [-0.01329461, -0.02031615, 0.01762123],
                            [0.01470665, 0.01854289, 0.01843571],
                            [-0.02344855, 0.00723628, -0.01725693],
                        ],
                        [
                            [0.00796995, -0.02797597, 0.00733656],
                            [-0.01105009, -0.00621813, -0.02718879],
                            [-0.00927974, -0.01958591, 0.02074316],
                            [-0.01964515, 0.01527182, -0.01675827],
                            [-0.01851097, 0.0215803, 0.00957259],
                            [0.02018503, 0.00962468, -0.01999826],
                            [-0.0153828, 0.01394675, -0.02165312],
                            [0.02233025, -0.01916396, -0.00583974],
                        ],
                    ]
                ),
            )
        else:
            raise KeyError(f'Unknown structure_id="{structure_id}"')
        return displacements

    return _generate_displacements


@pytest.fixture
def generate_displacement_dataset():
    """Return a `Dict` of displacement dataset of NaCl."""

    def _generate_displacement_dataset(structure_id="NaCl"):
        from aiida.orm import Dict

        if structure_id == "NaCl":
            dataset = Dict(
                dict={
                    "natom": 8,
                    "first_atoms": [
                        {"number": 0, "displacement": [0.03, 0.0, 0.0]},
                        {"number": 4, "displacement": [0.03, 0.0, 0.0]},
                    ],
                }
            )
        else:
            raise KeyError(f'Unknown structure_id="{structure_id}"')
        return dataset

    return _generate_displacement_dataset


@pytest.fixture
def generate_force_sets():
    """Return a `ArrayData` of force sets of NaCl."""

    def _generate_force_sets(structure_id="NaCl"):
        from aiida.orm import ArrayData

        if structure_id == "NaCl":
            force_sets = ArrayData()
            force_sets.set_array(
                "force_sets",
                np.array(
                    [
                        [
                            [-0.04527346, 0.0, 0.0],
                            [-0.00208978, 0.0, 0.0],
                            [0.00575753, 0.0, 0.0],
                            [0.00575753, 0.0, 0.0],
                            [-0.00179103, 0.0, 0.0],
                            [0.02865135, 0.0, 0.0],
                            [0.00449393, 0.0, 0.0],
                            [0.00449393, 0.0, 0.0],
                        ],
                        [
                            [-0.00159392, 0.0, 0.0],
                            [0.0288482, 0.0, 0.0],
                            [0.00468471, 0.0, 0.0],
                            [0.00468471, 0.0, 0.0],
                            [-0.0661841, 0.0, 0.0],
                            [-0.00333842, 0.0, 0.0],
                            [0.01644941, 0.0, 0.0],
                            [0.01644941, 0.0, 0.0],
                        ],
                    ]
                ),
            )
        elif structure_id == "NaCl-displacements":
            force_sets = ArrayData()
            force_sets.set_array(
                "force_sets",
                np.array(
                    [
                        [
                            [0.0372046, 0.00683762, 0.04328315],
                            [0.05577558, -0.0447563, -0.00060254],
                            [-0.0354772, -0.03504103, 0.00656763],
                            [-0.02014451, -0.01120216, 0.03760752],
                            [-0.06493306, 0.05667851, 0.02561049],
                            [-0.02701604, -0.02268542, -0.05351923],
                            [0.03333123, 0.03537694, -0.02011715],
                            [0.02125939, 0.01479183, -0.03882988],
                        ],
                        [
                            [0.03798966, 0.02499122, 0.00160333],
                            [-0.02015968, -0.02819268, 0.04410857],
                            [-0.04004788, 0.02353561, 0.03009268],
                            [0.00286866, -0.02823368, -0.04752076],
                            [-0.0339551, 0.01639552, 0.06342865],
                            [-0.09051561, -0.04241431, -0.02517307],
                            [0.05613445, 0.02843392, -0.03983418],
                            [0.08768549, 0.0054844, -0.02670521],
                        ],
                        [
                            [0.01579788, -0.02323645, 0.01300763],
                            [0.00769028, 0.06078973, 0.01203152],
                            [-0.06076039, -0.05093182, -0.01198282],
                            [-0.01915633, -0.00037008, 0.06857963],
                            [0.01495993, 0.03397984, -0.05208676],
                            [0.01675177, 0.04663717, -0.02782852],
                            [-0.02865451, -0.03255433, -0.04654033],
                            [0.05337138, -0.03431405, 0.04481964],
                        ],
                        [
                            [0.0053879, 0.05618248, -0.02316455],
                            [-0.00772245, -0.01453432, 0.01825674],
                            [0.03679958, 0.05113131, -0.04797775],
                            [0.01337501, -0.01495624, 0.02563117],
                            [0.0273215, -0.07028789, -0.06077977],
                            [-0.03444014, 0.00828853, 0.06870305],
                            [0.01396094, -0.06270262, 0.02340082],
                            [-0.05468234, 0.04687875, -0.00406971],
                        ],
                    ]
                ),
            )
        else:
            raise KeyError(f'Unknown structure_id="{structure_id}"')
        return force_sets

    return _generate_force_sets


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


@pytest.fixture
def generate_inputs_phonopy_wc(
    fixture_code,
    generate_structure,
    generate_displacement_dataset,
    generate_force_sets,
    generate_nac_params,
    generate_phonopy_settings,
):
    """Return inputs for phonopy workchain."""

    def _generate_inputs_phonopy(metadata=None):
        return {
            "code": fixture_code("phonoxpy.phonopy"),
            "structure": generate_structure(),
            "settings": generate_phonopy_settings(),
            "metadata": metadata or {},
            "force_sets": generate_force_sets(),
            "displacement_dataset": generate_displacement_dataset(),
            "nac_params": generate_nac_params(),
        }

    return _generate_inputs_phonopy
