# Phonopy workchain

Phonopy workchain is used to calculate force constants. Phonons and some phonon
derived properties can be optionally calculated.

The current usage is explained using python scripts found in the `examples`
directory. The first script is used to launch an aiida-phonopy calculation of
rocksalt NaCl.

```python
import copy

from aiida import load_profile
from aiida.engine import submit
from aiida.orm import Bool, Code, Float, Str, Dict, load_group, load_node
from aiida.plugins import DataFactory, WorkflowFactory
from aiida_phonoxpy.common.utils import phonopy_atoms_to_structure
from phonopy.interface.vasp import read_vasp_from_strings

load_profile()


def launch_aiida_vasp():
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

    supercell_matrix = [2, 2, 2]
    cutoff_energy = 350

    base_incar_dict = {
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
    }

    resources = {"parallel_env": "mpi*", "tot_num_mpiprocs": 24}
    kpoints_mesh = [2, 2, 2]
    base_parser_settings = {
        "add_energies": True,
        "add_forces": True,
        "add_stress": True,
    }
    code = Code.get_from_string("vasp621mpi@nancy")
    forces_config = {
        "code": code,
        # 'kpoints_density': 0.5,  # k-point density,
        "kpoints_mesh": kpoints_mesh,
        "kpoints_offset": [0.5, 0.5, 0.5],
        "potential_family": "PBE.54",
        "potential_mapping": {"Na": "Na_pv", "Cl": "Cl"},
        "options": {"resources": resources, "max_wallclock_seconds": 3600 * 10},
    }
    forces_incar_dict = base_incar_dict.copy()
    forces_incar_dict["NPAR"] = 4
    forces_config.update(
        {
            "settings": {"parser_settings": base_parser_settings},
            "parameters": {"incar": forces_incar_dict},
        }
    )
    nac_config = {
        "code": code,
        # 'kpoints_density': 0.5,  # k-point density,
        "kpoints_mesh": kpoints_mesh,
        "kpoints_offset": [0.5, 0.5, 0.5],
        "potential_family": "PBE.54",
        "potential_mapping": {"Na": "Na_pv", "Cl": "Cl"},
        "options": {"resources": resources, "max_wallclock_seconds": 3600 * 10},
    }
    nac_config["options"]["resources"]["tot_num_mpiprocs"] = 24
    nac_parser_settings = {"add_born_charges": True, "add_dielectrics": True}
    nac_parser_settings.update(base_parser_settings)
    nac_incar_dict = {"lepsilon": True}
    nac_incar_dict.update(base_incar_dict)
    nac_config.update(
        {
            "settings": {"parser_settings": nac_parser_settings},
            "parameters": {"incar": nac_incar_dict},
        }
    )

    builder = WorkflowFactory("phonoxpy.phonopy").get_builder()
    builder.structure = structure
    builder.calculator_inputs.force = forces_config
    builder.calculator_inputs.nac = nac_config
    builder.run_phonopy = Bool(True)
    builder.remote_phonopy = Bool(True)
    builder.code_string = Str("phonopy@nancy")
    phonon_settings = {
        "mesh": 50.0,
        "supercell_matrix": supercell_matrix,
        "distance": 0.03,
        # "number_of_snapshots": 4,
    }
    builder.settings = Dict(dict=phonon_settings)
    builder.symmetry_tolerance = Float(1e-5)
    label = "NaCl phonopy %dx%dx%d kpt %dx%dx%d PBE %d eV " "(distance :%4.2f)" % (
        tuple(supercell_matrix)
        + tuple(kpoints_mesh)
        + (
            cutoff_energy,
            phonon_settings["distance"],
        )
    )
    builder.metadata.label = label
    builder.metadata.description = label
    builder.phonopy.metadata.options.update(forces_config["options"])

    future = submit(builder)
    print(future)
    print("Running workchain with pk={}".format(future.pk))


def launch_aiida_qe():
    Dict = DataFactory("dict")
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

    supercell_matrix = [2, 2, 2]
    cutoff_energy = 350
    resources = {"parallel_env": "mpi*", "tot_num_mpiprocs": 24}
    options = {"resources": resources, "max_wallclock_seconds": 3600 * 10}
    kpoints_mesh = [2, 2, 2]
    base_parameters = {
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
    base_config = {
        "kpoints_mesh": kpoints_mesh,
        "kpoints_offset": [0.5, 0.5, 0.5],
        "pw": {
            "code_string": "qe-pw-6.8@nancy",
            "metadata": {"options": options},
            "pseudo_family_string": "SSSP/1.1/PBE/efficiency",
            "parameters": base_parameters,
        },
    }
    forces_config = copy.deepcopy(base_config)
    ph_parameters = {
        "INPUTPH": {
            "tr2_ph": 1.0e-8,
            "epsil": True,
        }
    }
    pw_config = copy.deepcopy(base_config)
    pw_config.update({"kpoints_mesh": [4, 4, 4], "kpoints_offset": [0, 0, 0]})
    ph_config = {
        "ph": {
            "code_string": "qe-ph-6.8@nancy",
            "metadata": {"options": options},
            "parameters": ph_parameters,
        }
    }
    nac_config = {"steps": [pw_config, ph_config]}
    builder = WorkflowFactory("phonoxpy.phonopy").get_builder()
    builder.structure = structure
    builder.calculator_inputs.force = forces_config
    builder.calculator_inputs.nac = nac_config
    builder.run_phonopy = Bool(True)
    builder.remote_phonopy = Bool(True)
    builder.code_string = Str("phonopy@nancy")
    phonon_settings = {
        "mesh": 50.0,
        "supercell_matrix": supercell_matrix,
        "distance": 0.03,
    }
    builder.settings = Dict(dict=phonon_settings)
    builder.symmetry_tolerance = Float(1e-5)
    builder.phonopy.metadata.options = options
    label = "NaCl qe-phonopy %dx%dx%d kpt %dx%dx%d PBE %d eV " "(distance :%4.2f)" % (
        tuple(supercell_matrix)
        + tuple(kpoints_mesh)
        + (
            cutoff_energy,
            phonon_settings["distance"],
        )
    )
    builder.metadata.label = label
    builder.metadata.description = label

    future = submit(builder)
    print(future)
    print("Running workchain with pk={}".format(future.pk))
```
