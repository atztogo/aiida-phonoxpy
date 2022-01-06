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
        elif structure_id == "NaCl-fc3":
            force_sets = ArrayData()
            force_sets.set_array(
                "force_sets",
                np.array(
                    [
                        [
                            [-4.5308910e-02, 0.0000000e00, -0.0000000e00],
                            [-2.0911800e-03, 0.0000000e00, 0.0000000e00],
                            [5.7499400e-03, 0.0000000e00, 0.0000000e00],
                            [5.7499400e-03, -0.0000000e00, 0.0000000e00],
                            [-1.7918500e-03, 0.0000000e00, 0.0000000e00],
                            [2.8618510e-02, 0.0000000e00, 0.0000000e00],
                            [4.5367800e-03, 0.0000000e00, 0.0000000e00],
                            [4.5367800e-03, 0.0000000e00, 0.0000000e00],
                        ],
                        [
                            [-1.5976600e-03, 0.0000000e00, -0.0000000e00],
                            [2.8796810e-02, 0.0000000e00, -0.0000000e00],
                            [4.7319400e-03, 0.0000000e00, -0.0000000e00],
                            [4.7319400e-03, 0.0000000e00, -0.0000000e00],
                            [-6.6104920e-02, -0.0000000e00, 0.0000000e00],
                            [-3.3256600e-03, 0.0000000e00, -0.0000000e00],
                            [1.6383770e-02, 0.0000000e00, -0.0000000e00],
                            [1.6383770e-02, 0.0000000e00, -0.0000000e00],
                        ],
                        [
                            [-8.0665190e-02, -3.1034070e-02, 0.0000000e00],
                            [-3.1733500e-03, 3.9472700e-03, 0.0000000e00],
                            [1.0214220e-02, -1.5987500e-03, 0.0000000e00],
                            [1.0222630e-02, 3.9654400e-03, 0.0000000e00],
                            [-2.6569300e-03, -1.3852700e-03, 0.0000000e00],
                            [4.9812020e-02, 3.0109600e-03, 0.0000000e00],
                            [8.1138200e-03, 2.0013550e-02, 0.0000000e00],
                            [8.1327700e-03, 3.0808800e-03, 0.0000000e00],
                        ],
                        [
                            [-1.2775200e-02, 3.1505840e-02, 0.0000000e00],
                            [-6.7295000e-04, -3.9993400e-03, 0.0000000e00],
                            [1.6233000e-03, 1.5453100e-03, 0.0000000e00],
                            [1.6249900e-03, -3.9993800e-03, 0.0000000e00],
                            [-5.8671000e-04, 1.3354300e-03, 0.0000000e00],
                            [8.2682200e-03, -3.1359100e-03, 0.0000000e00],
                            [1.2656300e-03, -2.0102340e-02, 0.0000000e00],
                            [1.2527100e-03, -3.1496100e-03, 0.0000000e00],
                        ],
                        [
                            [-4.6881540e-02, 3.9726100e-03, 0.0000000e00],
                            [-3.3384410e-02, -3.1293120e-02, 0.0000000e00],
                            [9.7264300e-03, 3.9742700e-03, 0.0000000e00],
                            [9.7234100e-03, -1.5725900e-03, 0.0000000e00],
                            [1.8269720e-02, 3.1026800e-03, 0.0000000e00],
                            [2.7252450e-02, -1.3620000e-03, 0.0000000e00],
                            [7.6543100e-03, 3.1123700e-03, 0.0000000e00],
                            [7.6396300e-03, 2.0065800e-02, 0.0000000e00],
                        ],
                        [
                            [-4.3738720e-02, -3.9685600e-03, 0.0000000e00],
                            [2.9203820e-02, 3.1289240e-02, 0.0000000e00],
                            [1.7754000e-03, -3.9746100e-03, 0.0000000e00],
                            [1.7787600e-03, 1.5727300e-03, 0.0000000e00],
                            [-2.1859840e-02, -3.1013000e-03, 0.0000000e00],
                            [2.9981930e-02, 1.3636500e-03, 0.0000000e00],
                            [1.4222300e-03, -3.1143800e-03, 0.0000000e00],
                            [1.4364300e-03, -2.0066770e-02, 0.0000000e00],
                        ],
                        [
                            [-4.2092340e-02, -1.3107500e-03, 3.2158600e-03],
                            [1.1257200e-03, 3.2180900e-03, -1.3072700e-03],
                            [-1.9584030e-02, -2.5336400e-02, -2.5348910e-02],
                            [4.4374700e-03, 3.2177800e-03, 3.2213500e-03],
                            [7.0988000e-04, 1.6349140e-02, 2.5045900e-03],
                            [3.1122020e-02, 2.5046800e-03, 1.6345200e-02],
                            [3.3986000e-03, -1.1410700e-03, -1.1399100e-03],
                            [2.0882680e-02, 2.4985300e-03, 2.5090900e-03],
                        ],
                        [
                            [-4.8526530e-02, 1.3130800e-03, -3.2329900e-03],
                            [-5.3077100e-03, -3.2185700e-03, 1.3096200e-03],
                            [3.1088170e-02, 2.5333530e-02, 2.5359910e-02],
                            [7.0616000e-03, -3.2177000e-03, -3.2191100e-03],
                            [-4.3026900e-03, -1.6336760e-02, -2.5084200e-03],
                            [2.6107330e-02, -2.5088200e-03, -1.6336410e-02],
                            [5.6808300e-03, 1.1404700e-03, 1.1389200e-03],
                            [-1.1801000e-02, -2.5052500e-03, -2.5115200e-03],
                        ],
                        [
                            [-4.6424430e-02, -1.1120500e-03, 0.0000000e00],
                            [1.8219130e-02, 3.3446400e-03, 0.0000000e00],
                            [9.0967100e-03, 2.0310310e-02, 0.0000000e00],
                            [9.1114800e-03, 3.3634600e-03, 0.0000000e00],
                            [-4.8571900e-02, -4.6775630e-02, 0.0000000e00],
                            [2.6286790e-02, 1.1603520e-02, 0.0000000e00],
                            [1.6137640e-02, -2.3391500e-03, 0.0000000e00],
                            [1.6144580e-02, 1.1604900e-02, 0.0000000e00],
                        ],
                        [
                            [-4.4199930e-02, 1.1171400e-03, 0.0000000e00],
                            [-2.2405290e-02, -3.3446900e-03, 0.0000000e00],
                            [2.4029200e-03, -2.0311480e-02, 0.0000000e00],
                            [2.3850000e-03, -3.3612600e-03, 0.0000000e00],
                            [4.4992300e-02, 4.6787480e-02, 0.0000000e00],
                            [3.0962250e-02, -1.1596440e-02, 0.0000000e00],
                            [-7.0686700e-03, 2.3331000e-03, 0.0000000e00],
                            [-7.0685900e-03, -1.1623860e-02, 0.0000000e00],
                        ],
                        [
                            [-2.4812240e-02, 3.3516000e-03, 0.0000000e00],
                            [-3.2255100e-03, -1.1127500e-03, 0.0000000e00],
                            [9.0922100e-03, 3.3640200e-03, 0.0000000e00],
                            [9.0728200e-03, 2.0312080e-02, 0.0000000e00],
                            [-4.1505400e-03, 1.1602360e-02, 0.0000000e00],
                            [-1.8215750e-02, -4.6780550e-02, 0.0000000e00],
                            [1.6124240e-02, 1.1605900e-02, 0.0000000e00],
                            [1.6114770e-02, -2.3426700e-03, 0.0000000e00],
                        ],
                        [
                            [-6.6158820e-02, -3.2844600e-03, 0.0000000e00],
                            [-9.7409000e-04, 1.1130200e-03, 0.0000000e00],
                            [2.3942600e-03, -3.3627700e-03, 0.0000000e00],
                            [2.4093200e-03, -2.0309170e-02, 0.0000000e00],
                            [5.4452000e-04, -1.1589770e-02, 0.0000000e00],
                            [7.5894810e-02, 4.6719330e-02, 0.0000000e00],
                            [-7.0623400e-03, -1.1609750e-02, 0.0000000e00],
                            [-7.0476700e-03, 2.3235600e-03, 0.0000000e00],
                        ],
                        [
                            [-4.2552310e-02, 1.6569540e-02, 2.7340400e-03],
                            [6.4820000e-04, 2.7370100e-03, 1.6566610e-02],
                            [4.8440000e-03, -9.0756000e-04, -9.0795000e-04],
                            [2.2318060e-02, 2.7392700e-03, 2.7386000e-03],
                            [7.6838800e-03, -1.9067200e-03, 9.4782400e-03],
                            [3.8092540e-02, 9.4787200e-03, -1.9061200e-03],
                            [-3.3662270e-02, -3.8190150e-02, -3.8181410e-02],
                            [2.6278900e-03, 9.4798900e-03, 9.4780000e-03],
                        ],
                        [
                            [-4.8033380e-02, -1.6526760e-02, -2.7433200e-03],
                            [-4.8281700e-03, -2.7369400e-03, -1.6565370e-02],
                            [6.6588100e-03, 9.0862000e-04, 9.0849000e-04],
                            [-1.0813010e-02, -2.7377200e-03, -2.7382800e-03],
                            [-1.1271180e-02, 1.9059700e-03, -9.4740100e-03],
                            [1.9138950e-02, -9.4708700e-03, 1.9068700e-03],
                            [4.2706760e-02, 3.8120270e-02, 3.8183210e-02],
                            [6.4412100e-03, -9.4625500e-03, -9.4776000e-03],
                        ],
                        [
                            [-3.2891910e-02, -3.1294360e-02, 0.0000000e00],
                            [2.7223210e-02, 3.9707800e-03, 0.0000000e00],
                            [8.7067800e-03, -1.5748700e-03, 0.0000000e00],
                            [8.7085000e-03, 3.9735700e-03, 0.0000000e00],
                            [-6.7456400e-02, -1.3501700e-03, 0.0000000e00],
                            [1.6741270e-02, 3.0983100e-03, 0.0000000e00],
                            [1.9482560e-02, 2.0063970e-02, 0.0000000e00],
                            [1.9486000e-02, 3.1127600e-03, 0.0000000e00],
                        ],
                        [
                            [2.9693900e-02, 3.1288690e-02, 0.0000000e00],
                            [3.0369000e-02, -3.9734000e-03, 0.0000000e00],
                            [7.6073000e-04, 1.5706300e-03, 0.0000000e00],
                            [7.5524000e-04, -3.9765500e-03, 0.0000000e00],
                            [-6.4731650e-02, 1.3646300e-03, 0.0000000e00],
                            [-2.3403230e-02, -3.0904400e-03, 0.0000000e00],
                            [1.3286150e-02, -2.0071760e-02, 0.0000000e00],
                            [1.3269860e-02, -3.1118100e-03, 0.0000000e00],
                        ],
                        [
                            [-3.1899900e-03, 3.9677000e-03, 0.0000000e00],
                            [-2.5349500e-03, -3.1298320e-02, 0.0000000e00],
                            [8.6911500e-03, 3.9717400e-03, 0.0000000e00],
                            [8.6850600e-03, -1.5791600e-03, 0.0000000e00],
                            [-4.5872430e-02, 3.0846300e-03, 0.0000000e00],
                            [-4.7092600e-03, -1.3227700e-03, 0.0000000e00],
                            [1.9486900e-02, 3.0911700e-03, 0.0000000e00],
                            [1.9443520e-02, 2.0085010e-02, 0.0000000e00],
                        ],
                        [
                            [-1.9020000e-05, -3.9698800e-03, 0.0000000e00],
                            [6.0603570e-02, 3.1230610e-02, 0.0000000e00],
                            [7.6141000e-04, -3.9707000e-03, 0.0000000e00],
                            [7.6038000e-04, 1.5738900e-03, 0.0000000e00],
                            [-8.6685310e-02, -3.0372900e-03, 0.0000000e00],
                            [-1.9813000e-03, 1.3819700e-03, 0.0000000e00],
                            [1.3301870e-02, -3.1031800e-03, 0.0000000e00],
                            [1.3258390e-02, -2.0105420e-02, 0.0000000e00],
                        ],
                        [
                            [1.6179300e-03, -1.3106100e-03, 3.2187500e-03],
                            [3.2012250e-02, 3.2166600e-03, -1.3090800e-03],
                            [-2.0618130e-02, -2.5340370e-02, -2.5353190e-02],
                            [3.4188800e-03, 3.2182700e-03, 3.2192600e-03],
                            [-6.3570850e-02, 1.6340680e-02, 2.5144600e-03],
                            [-8.1970000e-04, 2.5088100e-03, 1.6339000e-02],
                            [1.5241730e-02, -1.1402500e-03, -1.1388000e-03],
                            [3.2717900e-02, 2.5068100e-03, 2.5096100e-03],
                        ],
                        [
                            [-4.8165300e-03, 1.3130500e-03, -3.2197900e-03],
                            [2.5575800e-02, -3.2164900e-03, 1.3096300e-03],
                            [3.0058270e-02, 2.5283500e-02, 2.5354520e-02],
                            [6.0431100e-03, -3.2169000e-03, -3.2205000e-03],
                            [-6.8597720e-02, -1.6299110e-02, -2.5100200e-03],
                            [-5.8381700e-03, -2.5026000e-03, -1.6341420e-02],
                            [1.7533250e-02, 1.1443600e-03, 1.1358700e-03],
                            [4.1990000e-05, -2.5058200e-03, -2.5083000e-03],
                        ],
                        [
                            [-2.8619700e-03, -1.1288300e-03, 0.0000000e00],
                            [4.9523490e-02, 3.2472600e-03, 0.0000000e00],
                            [7.9056100e-03, 2.0240760e-02, 0.0000000e00],
                            [7.9448700e-03, 3.3484600e-03, 0.0000000e00],
                            [-1.1237686e-01, -4.6571910e-02, 0.0000000e00],
                            [-5.8232300e-03, 1.1595570e-02, 0.0000000e00],
                            [2.7850010e-02, -2.3511100e-03, 0.0000000e00],
                            [2.7838070e-02, 1.1619800e-02, 0.0000000e00],
                        ],
                        [
                            [-4.3556000e-04, 1.0932100e-03, 0.0000000e00],
                            [8.4192900e-03, -3.3820600e-03, 0.0000000e00],
                            [1.4106100e-03, -2.0340050e-02, 0.0000000e00],
                            [1.4167000e-03, -3.3835600e-03, 0.0000000e00],
                            [-1.9506160e-02, 4.6911200e-02, 0.0000000e00],
                            [-9.7000000e-04, -1.1587440e-02, 0.0000000e00],
                            [4.8466000e-03, 2.3155500e-03, 0.0000000e00],
                            [4.8185200e-03, -1.1626850e-02, 0.0000000e00],
                        ],
                        [
                            [1.8716510e-02, 3.3442300e-03, 0.0000000e00],
                            [2.7687510e-02, -1.1148800e-03, 0.0000000e00],
                            [8.1017100e-03, 3.3616500e-03, 0.0000000e00],
                            [8.0817800e-03, 2.0308170e-02, 0.0000000e00],
                            [-6.8444080e-02, 1.1594950e-02, 0.0000000e00],
                            [-5.0124660e-02, -4.6766060e-02, 0.0000000e00],
                            [2.8012810e-02, 1.1609590e-02, 0.0000000e00],
                            [2.7968420e-02, -2.3376500e-03, 0.0000000e00],
                        ],
                        [
                            [-2.1908340e-02, -3.3479400e-03, 0.0000000e00],
                            [2.9913240e-02, 1.1112900e-03, 0.0000000e00],
                            [1.3721400e-03, -3.3660100e-03, 0.0000000e00],
                            [1.3866200e-03, -2.0312370e-02, 0.0000000e00],
                            [-6.3780020e-02, -1.1598660e-02, 0.0000000e00],
                            [4.3445180e-02, 4.6787590e-02, 0.0000000e00],
                            [4.7954700e-03, -1.1603960e-02, 0.0000000e00],
                            [4.7757200e-03, 2.3300500e-03, 0.0000000e00],
                        ],
                        [
                            [1.1389700e-03, 1.6566450e-02, 2.7378800e-03],
                            [3.1532850e-02, 2.7374200e-03, 1.6564410e-02],
                            [3.8236700e-03, -9.0804000e-04, -9.0842000e-04],
                            [2.1297940e-02, 2.7391500e-03, 2.7392500e-03],
                            [-5.6620160e-02, -1.9054400e-03, 9.4775200e-03],
                            [6.1493500e-03, 9.4799900e-03, -1.9066200e-03],
                            [-2.1796900e-02, -3.8185250e-02, -3.8180710e-02],
                            [1.4474290e-02, 9.4757100e-03, 9.4767000e-03],
                        ],
                        [
                            [-4.3384300e-03, -1.6566400e-02, -2.7380500e-03],
                            [2.6054340e-02, -2.7386000e-03, -1.6567010e-02],
                            [5.6404500e-03, 9.0696000e-04, 9.0717000e-04],
                            [-1.1833970e-02, -2.7393200e-03, -2.7360100e-03],
                            [-7.5583580e-02, 1.9088800e-03, -9.4991300e-03],
                            [-1.2806640e-02, -9.4749900e-03, 1.9040100e-03],
                            [5.4579990e-02, 3.8180430e-02, 3.8204000e-02],
                            [1.8287830e-02, -9.4769600e-03, -9.4749800e-03],
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
    """Generate a `RemoteData` node."""

    def _generate_remote_data(computer, remote_path, entry_point_name=None):
        """Generate a `RemoteData` node."""
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

            from aiida_phonoxpy.utils.utils import phonopy_atoms_to_structure

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


@pytest.fixture
def mock_calculator_code():
    """Return Code instance mock that returns plugin name."""

    def _mock_code(plugin_name):
        class MockCode:
            @staticmethod
            def get_input_plugin_name():
                return plugin_name

        return MockCode()

    return _mock_code
