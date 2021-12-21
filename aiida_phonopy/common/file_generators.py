"""Collection of functions to generate phonopy/phono3py files."""
from phonopy import Phonopy
from phonopy.structure.dataset import forces_in_dataset
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.file_IO import (
    get_FORCE_SETS_lines,
    get_BORN_lines,
    get_FORCE_CONSTANTS_lines,
)
from aiida_phonopy.common.utils import phonopy_atoms_from_structure


def get_BORN_txt(nac_data, structure, symmetry_tolerance):
    """Return BORN file as text.

    nac_data : ArrayData
        Born effective charges and dielectric constants
    structure : StructureData
        This is assumed to be the primitive cell in workchain.
    symmetry_tolerance : float
        Symmetry tolerance.

    """
    born_charges = nac_data.get_array("born_charges")
    epsilon = nac_data.get_array("epsilon")
    pcell = phonopy_atoms_from_structure(structure)
    lines = get_BORN_lines(
        pcell, born_charges, epsilon, symprec=symmetry_tolerance.value
    )

    return "\n".join(lines)


def get_FORCE_SETS_txt(dataset, force_sets=None):
    """Return FORCE_SETS file as text."""
    if dataset is None:
        return None
    if force_sets is None and not forces_in_dataset(dataset):
        return None
    if dataset is not None and force_sets is not None:
        forces = force_sets.get_array("force_sets")
        lines = get_FORCE_SETS_lines(dataset, forces=forces)
    elif dataset is not None and force_sets is None:
        lines = get_FORCE_SETS_lines(dataset)

    return "\n".join(lines)


def get_FORCE_CONSTANTS_txt(force_constants_object):
    """Return FORCE_CONSTANTS file as text."""
    force_constants = force_constants_object.get_array("force_constants")
    p2s_map = force_constants_object.get_array("p2s_map")
    lines = get_FORCE_CONSTANTS_lines(force_constants, p2s_map=p2s_map)

    return "\n".join(lines)


def get_phonopy_yaml_txt(
    structure, supercell_matrix=None, primitive_matrix=None, calculator=None
):
    """Return phonopy.yaml file as text."""
    unitcell = phonopy_atoms_from_structure(structure)
    ph = Phonopy(
        unitcell,
        supercell_matrix=supercell_matrix,
        primitive_matrix="auto",
        calculator=calculator,
    )
    phpy_yaml = PhonopyYaml()
    phpy_yaml.set_phonon_info(ph)

    return str(phpy_yaml)


def get_phonopy_options(postprocess_parameters):
    """Return phonopy command options as strings."""
    mesh_opts = []
    if "mesh" in postprocess_parameters:
        mesh = postprocess_parameters["mesh"]
        try:
            length = float(mesh)
            mesh_opts.append("--mesh=%f" % length)
        except TypeError:
            mesh_opts.append('--mesh="%d %d %d"' % tuple(mesh))
        mesh_opts.append("--nowritemesh")

    fc_opts = []
    if "fc_calculator" in postprocess_parameters:
        if postprocess_parameters["fc_calculator"].lower().strip() == "alm":
            fc_opts.append("--alm")
    return mesh_opts, fc_opts
