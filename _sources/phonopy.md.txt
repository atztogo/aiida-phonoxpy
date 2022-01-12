# Phonopy workchain

Phonopy workchain is used to calculate harmonic phonon properties.

## Phonon calculation settings

Supported parameters for phonon calculation are

- `supercell_matrix`: Dimension of supercell used by phonopy
- `distance` (optional): Atomic displacement distance.
- `mesh` (optional): Sampling mesh used in phonon projected DOS and thermal
  property calculations.
- `fc_calculator` (optional): Only "alm" is supported.
- `number_of_snapshots` (optional): Random displacements.
- `random_seed` (optional): Random seed used to generate displacements.

Detailed explanations about the parameters are refereed to the
[phonopy documentation](https://phonopy.github.io/phonopy/).

An example of settings is:

```python
phonopy_settings = {
    "mesh": 50.0,
    "supercell_matrix": supercell_matrix,
    "distance": 0.03,
}
```

Without setting `number_of_snapshots`, systematic atomic displacements
considering crystal symmetry are introduced to perfect supercells. The
displacement information is stored in `displacement_dataset`. With setting
`number_of_snapshots`, all atomics are randomly displaced in direction at a
fixed displacement distance. In this case, the displacement information is
stored in `displacements`.

## Initialization

At the initialization step, supercells with displacements are created. Data
nodes of

- `displacement_dataset` or `displacements`
- `phonon_setting_info`

are created and attached to `outputs` port namespace.

## Calculations launched from Phonopy workchain

Specific calculations are launched from the phonopy workchain to obtain data
necessary for phonon calculations, which are intermediate data calculated by
external calculator, supercell forces and parameters for non-analytical term
correction (NAC). Using these intermediate data, phonon properties are
calculated. The former calculations are the computationally demanding part.
Therefore for the practical purpose, those data are stored in data nodes and
attached to `outputs` port namespace. With these data, phonon properties are
calculated and stored in data nodes and attached to `outputs` port namespace.
Since the later phonon calculation is much less computational demanding, it is
not performed as default.

- `force_sets`: Sets of supercell forces
- `nac_params`: Parameters for non-analytical term correction
- `force_constants`, `thermal_properties`, `band_structure`, `projected_dos`:
  Phonon properties for which `run_phonopy = Bool(True)` has to be set to
  perform the calculation

## Usage and examples

The basic usage is explained as a python script shown below.

- `launch_phonopy_with_vasp`: Run phonon calculation using phonopy & VASP.
- `launch_phonopy_with_qe`: Run phonon calculation using phonopy & QE.
- `launch_phonopy_with_precalculated_dataset`: Run phonon calculation using
  outputs of previous `PhonopyWorkChain` calculation without running supercell
  force and NAC calculations.

```python
"""Examples to submit PhonopyWorkChain."""
from aiida import load_profile
from aiida.engine import submit
from aiida.orm import Bool, Code, Dict, load_group, load_node
from aiida.plugins import WorkflowFactory
from aiida_phonoxpy.utils.utils import phonopy_atoms_to_structure
from phonopy.interface.vasp import read_vasp_from_strings

load_profile()


def launch_phonopy_with_precalculated_dataset():
    """Submit phonopy calculation with pre-calculated force_sets."""
    n = load_node(226097)  # PhonopyWorkchain by launch_phonopy_with_vasp.
    supercell_matrix = n.outputs.phonon_setting_info["supercell_matrix"]

    #
    # Settings for phonopy calculation
    #
    phonopy_settings = {
        "mesh": 100.0,
        "supercell_matrix": supercell_matrix,
    }
    phonopy_resources = {"parallel_env": "mpi*", "tot_num_mpiprocs": 24}
    phonopy_metadata = {
        "options": {
            "resources": phonopy_resources,
            "max_wallclock_seconds": 3600 * 10,
        },
    }

    #
    # PhonopyWorkChain inputs
    #
    PhonopyWorkChain = WorkflowFactory("phonoxpy.phonopy")
    label = f"NaCl phonopy calculation after PhonopyWorkChain (pk={n.pk})"
    inputs = {
        "code": Code.get_from_string("phonopy@nancy"),
        "structure": n.inputs.structure,
        "settings": Dict(dict=phonopy_settings),
        "force_sets": n.outputs.force_sets,
        "nac_params": n.outputs.nac_params,
        "displacement_dataset": n.outputs.displacement_dataset,
        "run_phonopy": Bool(True),
        "metadata": {
            "label": label,
            "description": label,
        },
        "phonopy": {"metadata": phonopy_metadata},
    }

    g = load_group("phonopy-example")
    node = submit(PhonopyWorkChain, **inputs)
    print(node)
    g.add_nodes(node)


def launch_phonopy_with_vasp():
    """Run PhonopyWorkChain using aiida-vasp."""
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
    phonopy_settings = {
        "mesh": 50.0,
        "supercell_matrix": supercell_matrix,
        "distance": 0.03,
    }
    phonopy_resources = {"parallel_env": "mpi*", "tot_num_mpiprocs": 24}
    phonopy_metadata = {
        "options": {
            "resources": phonopy_resources,
            "max_wallclock_seconds": 3600 * 10,
        },
    }

    #
    # PhonopyWorkChain inputs
    #
    PhonopyWorkChain = WorkflowFactory("phonoxpy.phonopy")
    label = "NaCl phonopy %dx%dx%d kpt %dx%dx%d PBE %d eV " "(distance :%4.2f)" % (
        tuple(supercell_matrix)
        + tuple(force_kpoints_mesh)
        + (
            cutoff_energy,
            phonopy_settings["distance"],
        )
    )
    inputs = {
        "structure": structure,
        "calculator_inputs": {"force": force_config, "nac": nac_config},
        "run_phonopy": Bool(True),
        "code": Code.get_from_string("phonopy@nancy"),
        "settings": Dict(dict=phonopy_settings),
        "metadata": {
            "label": label,
            "description": label,
        },
        "phonopy": {"metadata": phonopy_metadata},
    }

    g = load_group("phonopy-example")
    node = submit(PhonopyWorkChain, **inputs)
    print(node)
    g.add_nodes(node)


def launch_phonopy_with_qe():
    """Run PhonopyWorkChain using aiida-quantumespresso."""
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

    #
    # Settings for supercell forces
    #
    supercell_matrix = [2, 2, 2]
    force_kpoints_mesh = [2, 2, 2]
    force_resources = {"parallel_env": "mpi*", "tot_num_mpiprocs": 24}
    force_options = {"resources": force_resources, "max_wallclock_seconds": 3600 * 10}
    force_parameters = {
        "CONTROL": {
            "calculation": "scf",
            "restart_mode": "from_scratch",
            "tprnfor": True,
        },
        "SYSTEM": {
            "occupations": "fixed",
            "ecutwfc": 30.0,
            "ecutrho": 240.0,
        },
        "ELECTRONS": {
            "conv_thr": 1.0e-6,
        },
    }
    force_config = {
        "kpoints_mesh": force_kpoints_mesh,
        "kpoints_offset": [0.5, 0.5, 0.5],
        "pw": {
            "code_string": "qe-pw-6.8@nancy",
            "metadata": {"options": force_options},
            "pseudo_family_string": "SSSP/1.1/PBE/efficiency",
            "parameters": force_parameters,
        },
    }

    #
    # Settings for Born effective charges and dielectric constant.
    #
    nac_resources = {"parallel_env": "mpi*", "tot_num_mpiprocs": 24}
    nac_options = {"resources": nac_resources, "max_wallclock_seconds": 3600 * 10}
    nac_pw_parameters = {
        "CONTROL": {
            "calculation": "scf",
            "restart_mode": "from_scratch",
            "tprnfor": True,
        },
        "SYSTEM": {
            "occupations": "fixed",
            "ecutwfc": 30.0,
            "ecutrho": 240.0,
        },
        "ELECTRONS": {
            "conv_thr": 1.0e-6,
        },
    }
    nac_ph_parameters = {
        "INPUTPH": {
            "tr2_ph": 1.0e-8,
            "epsil": True,
        }
    }
    nac_config = {
        "steps": [
            {
                "kpoints_mesh": [8, 8, 8],
                "kpoints_offset": [0, 0, 0],
                "pw": {
                    "code_string": "qe-pw-6.8@nancy",
                    "metadata": {"options": nac_options},
                    "pseudo_family_string": "SSSP/1.1/PBE/efficiency",
                    "parameters": nac_pw_parameters,
                },
            },
            {
                "ph": {
                    "code_string": "qe-ph-6.8@nancy",
                    "metadata": {"options": nac_options},
                    "parameters": nac_ph_parameters,
                }
            },
        ],
    }

    #
    # Settings for phonopy calculation
    #
    phonopy_settings = {
        "mesh": 50.0,
        "supercell_matrix": supercell_matrix,
        "distance": 0.03,
    }
    phonopy_resources = {"parallel_env": "mpi*", "tot_num_mpiprocs": 24}
    phonopy_metadata = {
        "options": {
            "resources": phonopy_resources,
            "max_wallclock_seconds": 3600 * 10,
        },
    }

    #
    # PhonopyWorkChain inputs
    #
    PhonopyWorkChain = WorkflowFactory("phonoxpy.phonopy")
    label = "NaCl qe-phonopy %dx%dx%d kpt %dx%dx%d PBE" "(distance :%4.2f)" % (
        tuple(supercell_matrix)
        + tuple(force_kpoints_mesh)
        + (phonopy_settings["distance"],)
    )
    inputs = {
        "structure": structure,
        "calculator_inputs": {"force": force_config, "nac": nac_config},
        "run_phonopy": Bool(True),
        "code": Code.get_from_string("phonopy@nancy"),
        "settings": Dict(dict=phonopy_settings),
        "metadata": {
            "label": label,
            "description": label,
        },
        "phonopy": {"metadata": phonopy_metadata},
    }

    g = load_group("phonopy-example")
    node = submit(PhonopyWorkChain, **inputs)
    print(node)
    g.add_nodes(node)
```
