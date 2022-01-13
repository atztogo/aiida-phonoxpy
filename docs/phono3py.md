# Phono3py workchains

Before reading this section, it is recommended to read {ref}`phonooy_workchain`
and try `PhonopyWorkChain`because how to use`Phono3pyWorkChain`is similar to
that of`PhonopyWorkChain`and it is assumed you have already
experienced`PhonopyWorkChain`.

Phono3py workchain is used to calculate phonon-phonon interaction properties.
There are three phono3py workchains:

1. `Phono3pyWorkChain`: Calculate supercell forces, parameters for
   non-analytical term correction (NAC), and optionally run the following two
   workchains.
2. `Phono3pyFCWorkChain`: Calculate force constants from supercell forces and
   displacements.
3. `Phono3pyLTCWorkChain`: Calculate lattice thermal conductivity from force
   constants and NAC parameters.

## Phonon-phonon interaction calculation settings

Supported parameters for phonon calculation are

- `supercell_matrix`: Dimension of supercell used by phono3py.
- `phonon_supercell_matrix` (optional): Dimension of supercell used to calculate
  harmonic phonon by phono3py.
- `distance` (optional): Atomic displacement distance.
- `mesh` (optional): Sampling mesh used in phonon projected DOS and thermal
  property calculations.
- `fc_calculator` (optional): Only "alm" is supported.

Although random displacement setting (`number_of_snapshots`) is not directory
supported by `Phono3pyWorkChain`, supercell forces and displacements calculated
using random displacement setting by `PhonopyWorkChain` can be used as inputs of
`Phono3pyFCWorkChain`.

Detailed explanations about the parameters are refereed to the
[phono3py documentation](https://phonopy.github.io/phono3py/).

An example of settings is:

```python
phonopy_settings = {
    "mesh": 50.0,
    "supercell_matrix": [2, 2, 2],
    "phonon_supercell_matrix": [4, 4, 4]
    "distance": 0.03,
}
```

The displacement information is stored in `displacement_dataset`. With setting
`phonon_supercell_matrix`, the displacement information is stored in
`phonon_displacement_dataset`.

## Initialization

At the initialization step, supercells with displacements are created. Data
nodes of

- `displacement_dataset` and `phonon_displacement_dataset` (optional).
- `phonon_setting_info`

are created and attached to `outputs` port namespace. Input `structure` is
considered as a conventional unit cell, which can be obtained using, e.g., using
spglib. The primitive cell is found automatically and the transformation matrix
from the input `structure` to the primitive cell is found in
`phonon_setting_info` as `primitive_matrix`.

## Calculations launched from Phono3py workchain

1. Supercell forces and NAC parameters

   Similar to `PhonopyWorkChain` ({ref}`launch_calculators`),

   - `force_sets`
   - `nac_params` (optional)

   are obtained. In addition,

   - `phonon_force_sets` (optional)

   is obtained when `phonon_supercell_matrix` is set.

2. Force constants (when `run_fc = Bool(True)`)

   - `fc2`: `SingleFileData` of `fc2.hdf5`.
   - `fc3`: `SingleFileData` of `fc3.hdf5`.

3. Lattice thermal conductivity calculation (when `run_ltc = Bool(True)`)

   - `ltc`: `SingleFileData` of `kappa-xxx.hdf5`.

## Usage and examples

The basic usage is explained as a python script shown below.

- `launch_phono3py_with_vasp`: Run phono3py calculation using phono3py & VASP.
- `launch_phono3py_fc`: Run phono3py force constants calculation.
- `launch_phono3py_ltc`: Run phono3py lattice thermal conductivity calculation.

```python
"""Examples to submit Phono3pyWorkChain, Phono3pyFCWorkChain, Phono3pyLTCWorkChain."""
from aiida import load_profile
from aiida.engine import submit
from aiida.orm import Bool, Code, Dict, load_group, load_node
from aiida.plugins import WorkflowFactory
from aiida_phonoxpy.utils.utils import phonopy_atoms_to_structure
from phonopy.interface.vasp import read_vasp_from_strings

load_profile()


def launch_phonop3y_fc():
    """Submit phono3py force constants calculation."""
    # Supercell force calculations with random displacements.
    n222 = load_node(220964)  # PhonopyWorkchain with 2x2x2 supercell (200 supercells)
    n444 = load_node(220888)  # PhonopyWorkchain with 4x4x4 supercell (4 supercells)

    #
    # Settings for phono3py calculation
    #
    phono3py_settings = {
        "supercell_matrix": n222.outputs.phonon_setting_info["supercell_matrix"],
        "phonon_supercell_matrix": n444.outputs.phonon_setting_info["supercell_matrix"],
    }
    phono3py_resources = {"parallel_env": "mpi*", "tot_num_mpiprocs": 24}
    phono3py_metadata = {
        "options": {
            "resources": phono3py_resources,
            "max_wallclock_seconds": 3600 * 10,
        },
    }

    #
    # Phono3pyFCWorkChain inputs
    #
    label = "NaCl phono3py force constants calculation (222-444)"
    inputs = {
        "code": Code.get_from_string("phono3py@nancy"),
        "structure": n222.inputs.structure,
        "settings": Dict(dict=phono3py_settings),
        "force_sets": n222.outputs.force_sets,
        "displacements": n222.outputs.displacements,
        "phonon_force_sets": n444.outputs.force_sets,
        "phonon_displacements": n444.outputs.displacements,
        "metadata": {
            "label": label,
            "description": label,
        },
        "phono3py": {"metadata": phono3py_metadata},
    }

    g = load_group("NaCl-222-444")
    Phono3pyFCWorkChain = WorkflowFactory("phonoxpy.phono3py_fc")
    node = submit(Phono3pyFCWorkChain, **inputs)
    print(node)
    g.add_nodes(node)


def launch_phonop3y_ltc():
    """Submit phono3py force constants calculation."""
    n_fc = load_node(257252)  # Phono3pyFCWorkchain
    n_nac = load_node(255135)  # NacParamsWorkChain

    #
    # Settings for phono3py calculation
    #
    phono3py_settings = {
        "supercell_matrix": n_fc.outputs.phonon_setting_info["supercell_matrix"],
        "phonon_supercell_matrix": n_fc.outputs.phonon_setting_info[
            "phonon_supercell_matrix"
        ],
        "mesh": 100,
    }
    phono3py_resources = {"parallel_env": "mpi*", "tot_num_mpiprocs": 24}
    phono3py_metadata = {
        "options": {
            "resources": phono3py_resources,
            "max_wallclock_seconds": 3600 * 10,
        },
    }

    #
    # Phono3pyFCWorkChain inputs
    #
    label = "NaCl phono3py lattice thermal conductivity calculation (222-444)"
    inputs = {
        "code": Code.get_from_string("phono3py@nancy"),
        "structure": n_fc.inputs.structure,
        "settings": Dict(dict=phono3py_settings),
        "fc2": n_fc.outputs.fc2,
        "fc3": n_fc.outputs.fc3,
        "nac_params": n_nac.outputs.nac_params,
        "metadata": {
            "label": label,
            "description": label,
        },
        "phono3py": {"metadata": phono3py_metadata},
    }

    g = load_group("NaCl-222-444")
    Phono3pyLTCWorkChain = WorkflowFactory("phonoxpy.phono3py_ltc")
    node = submit(Phono3pyLTCWorkChain, **inputs)
    print(node)
    g.add_nodes(node)


def launch_phono3py_with_vasp():
    """Run Phono3pyWorkChain using aiida-vasp."""
    unitcell_str = """ Na Cl
   1.00000000000000
     5.6903014761756712    0.0000000000000000    0.0000000000000000
     0.0000000000000000    5.6903014761756712    0.0000000000000000
     0.0000000000000000    0.0000000000000000    5.6903014761756712
   4   4
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000  0.5000000000000000  0.5000000000000000
  0.5000000000000000  0.0000000000000000  0.5000000000000000
  0.5000000000000000  0.5000000000000000  0.0000000000000000
  0.5000000000000000  0.5000000000000000  0.5000000000000000
  0.5000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000  0.5000000000000000  0.0000000000000000
  0.0000000000000000  0.0000000000000000  0.5000000000000000"""

    cell = read_vasp_from_strings(unitcell_str)
    structure = phonopy_atoms_to_structure(cell)

    cutoff_energy = 350

    #
    # Settings for supercell forces
    #
    supercell_matrix = [2, 2, 2]
    force_kpoints_mesh = [2, 2, 2]
    force_incar_dict = {
        "PREC": "Accurate",
        "IBRION": -1,
        "EDIFF": 1e-8,
        "NELMIN": 5,
        "NELM": 100,
        "ENCUT": cutoff_energy,
        "IALGO": 38,
        "ISMEAR": 0,
        "SIGMA": 0.01,
        "LREAL": False,
        "lcharg": False,
        "lwave": False,
        "NPAR": 4,
    }
    force_code = Code.get_from_string("vasp621mpi@nancy")
    force_resources = {"parallel_env": "mpi*", "tot_num_mpiprocs": 24}

    force_parser_settings = {
        "add_energies": True,
        "add_forces": True,
        "add_stress": True,
    }
    force_config = {
        "code": force_code,
        "kpoints_mesh": force_kpoints_mesh,
        "kpoints_offset": [0.5, 0.5, 0.5],
        "potential_family": "PBE.54",
        "potential_mapping": {"Na": "Na_pv", "Cl": "Cl"},
        "options": {
            "resources": force_resources,
            "max_wallclock_seconds": 3600 * 10,
        },
    }
    force_config.update(
        {
            "settings": {"parser_settings": force_parser_settings},
            "parameters": {"incar": force_incar_dict},
        }
    )

    #
    # Settings for phonon supercell forces
    #
    phonon_supercell_matrix = [4, 4, 4]
    phonon_force_kpoints_mesh = [1, 1, 1]
    phonon_force_incar_dict = {
        "PREC": "Accurate",
        "IBRION": -1,
        "EDIFF": 1e-8,
        "NELMIN": 5,
        "NELM": 100,
        "ENCUT": cutoff_energy,
        "IALGO": 48,
        "ISMEAR": 0,
        "SIGMA": 0.01,
        "LREAL": False,
        "lcharg": False,
        "lwave": False,
        "NPAR": 8,
    }
    phonon_force_code = Code.get_from_string("vasp621mpi@asahi")
    phonon_force_resources = {"num_machines": 2, "tot_num_mpiprocs": 96}

    phonon_force_parser_settings = {
        "add_energies": True,
        "add_forces": True,
        "add_stress": True,
    }
    phonon_force_config = {
        "code": phonon_force_code,
        "kpoints_mesh": phonon_force_kpoints_mesh,
        "kpoints_offset": [0.5, 0.5, 0.5],
        "potential_family": "PBE.54",
        "potential_mapping": {"Na": "Na_pv", "Cl": "Cl"},
        "options": {
            # "queue_name": "qM048",
            "resources": phonon_force_resources,
            "max_wallclock_seconds": 3600 * 10,
        },
    }
    phonon_force_config.update(
        {
            "settings": {"parser_settings": phonon_force_parser_settings},
            "parameters": {"incar": phonon_force_incar_dict},
        }
    )

    #
    # Settings for Born effective charges and dielectric constant.
    #
    nac_code = Code.get_from_string("vasp621mpi@nancy")
    nac_resources = {"parallel_env": "mpi*", "tot_num_mpiprocs": 24}
    nac_config = {
        "code": nac_code,
        "kpoints_mesh": [8, 8, 8],
        "kpoints_offset": [0, 0, 0],
        "potential_family": "PBE.54",
        "potential_mapping": {"Na": "Na_pv", "Cl": "Cl"},
        "options": {"resources": nac_resources, "max_wallclock_seconds": 3600 * 10},
    }
    nac_parser_settings = {
        "add_energies": True,
        "add_forces": True,
        "add_stress": True,
        "add_born_charges": True,
        "add_dielectrics": True,
    }
    nac_incar_dict = {
        "PREC": "Accurate",
        "IBRION": -1,
        "EDIFF": 1e-8,
        "NELMIN": 5,
        "NELM": 100,
        "ENCUT": cutoff_energy,
        "IALGO": 38,
        "ISMEAR": 0,
        "SIGMA": 0.01,
        "LREAL": False,
        "lcharg": False,
        "lwave": False,
        "lepsilon": True,
    }
    nac_config.update(
        {
            "settings": {"parser_settings": nac_parser_settings},
            "parameters": {"incar": nac_incar_dict},
        }
    )

    #
    # Settings for phonopy calculation
    #
    phono3py_settings = {
        "mesh": 50.0,
        "supercell_matrix": supercell_matrix,
        "phonon_supercell_matrix": phonon_supercell_matrix,
        "distance": 0.03,
    }
    phono3py_resources = {"parallel_env": "mpi*", "tot_num_mpiprocs": 24}
    phono3py_metadata = {
        "options": {
            "resources": phono3py_resources,
            "max_wallclock_seconds": 3600 * 10,
        },
    }

    #
    # Phono3pyWorkChain inputs
    #
    label = "NaCl phonopy %dx%dx%d kpt %dx%dx%d PBE %d eV (distance :%4.2f)" % (
        tuple(supercell_matrix)
        + tuple(force_kpoints_mesh)
        + (
            cutoff_energy,
            phono3py_settings["distance"],
        )
    )
    inputs = {
        "structure": structure,
        "calculator_inputs": {
            "force": force_config,
            "nac": nac_config,
            "phonon_force": phonon_force_config,
        },
        "run_fc": Bool(True),
        "run_ltc": Bool(True),
        "code": Code.get_from_string("phono3py@nancy"),
        "settings": Dict(dict=phono3py_settings),
        "metadata": {
            "label": label,
            "description": label,
        },
        "phono3py": {"metadata": phono3py_metadata},
    }

    g = load_group("phono3py-example")
    Phono3pyWorkChain = WorkflowFactory("phonoxpy.phono3py")
    node = submit(Phono3pyWorkChain, **inputs)
    print(node)
    g.add_nodes(node)
```
