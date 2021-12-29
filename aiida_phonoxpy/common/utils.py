"""General utilities."""

from typing import Optional
import numpy as np
from aiida.common import InputValidationError
from aiida.engine import calcfunction
from aiida.orm import (
    Bool,
    Float,
    Dict,
    StructureData,
    ArrayData,
    XyData,
    BandsData,
    KpointsData,
)
from phonopy import Phonopy
from phonopy.interface.calculator import get_default_physical_units
from phonopy.structure.atoms import PhonopyAtoms


@calcfunction
def get_remote_fc_calculation_settings(phonon_settings: Dict):
    """Create remote force constants phonopy calculation setting.

    keys condidered:
        supercell_matrix
        fc_calculator

    """
    key = "supercell_matrix"
    if key in phonon_settings.keys():
        fc_settings = {key: phonon_settings[key]}
    else:
        return None
    key = "fc_calculator"
    if key in phonon_settings.dict:
        fc_settings["postprocess_parameters"] = {key: phonon_settings[key]}
    return Dict(dict=fc_settings)


@calcfunction
def setup_phonopy_calculation(
    phonon_settings: Dict,
    structure: StructureData,
    symmetry_tolerance: Float,
    run_phonopy: Bool,
    displacement_dataset: Optional[Dict] = None,
    displacements: Optional[ArrayData] = None,
):
    """Set up phonopy calculation.

    Valid keys in phonon_settings_info are
        ('supercell_matrix',
         'phonon_supercell_matrix',
         'distance',
         'symmetry_tolerance',
         'number_of_snapshots',
         'random_seed',
         'is_plusminus',
         'is_diagonal',
         'is_trigonal',
         'mesh',
         'fc_calculator').

    Returns
    -------
    dict
        'supercell' : StructureData
            Perfect supercell.
        'supercell_001', 'supercell_002', ... : StructureData
            Supercells with displacements
        'primitive' : StructureData
            Primitive cell.
        'phonon_supercell' : StructureData
            For phono3py. Perfect supercell for harmonic phonon calculation.
        'phonon_supercell_001', 'phonon_supercell_002', ... : StructureData
            For phono3py. Supercells with displacements for harmonic phonon
            calculation.
        'phonon_setting_info' : Dict
            Phonopy setting parameters including those generated in the
            process of displacements creation, e.g., primitive and  sueprcell
            matrix and symmetry information.
        'displacement_dataset' : Dict, optional
            When 'number_of_snapshots' is not given, displacements are generated
            in a usual way (least number of displacements considering symmetry),
            and phonopy's type-I displacement dataset is returned.
        'displacements' : ArrayData, optional
            When 'number_of_snapshots' is given, random displacements are
            generated and phonopy's type-II displacements array is returned.

    phonon_setting_info contains the following entries:
        'version' : str
            Phonopy version number.
        'supercell_matrix' : array_like
            3x3 integer matrix to generate supercell matrix.
        'phonon_supercell_matrix' : array_like
            3x3 integer matrix to generate fc2 supercell matrix for Phono3py.
        'distance' : float
            Displacement distance.
        'symmetry_tolerance' : float
            Tolerance length used for symmetry finding.
        'primitive_matrix' : array_like
            Phonopy.primitive_matrix.
        'symmetry' : dict
            'number' : Space group number.
            'international' : Space group type.
        'phonon_displacement_dataset' : dict
            Phono3py.phonon_dataset.
        'number_of_snapshots' : int
        'random_seed' : int
        'is_plusminus' : str or bool
        'is_diagonal' : bool
        'mesh' : float or list, optional
            Mesh numbers or distance measure of q-point sampling mesh.
        'fc_calculator' : str, optional
            External force constants calculator.

    """
    ph_settings = _get_setting_info(phonon_settings)
    if run_phonopy:
        params = _get_phonopy_postprocess_info(phonon_settings)
        ph_settings.update(params)

    ph = _get_phonopy_instance(
        structure, ph_settings, symmetry_tolerance=symmetry_tolerance.value
    )
    ph_settings["version"] = ph.version
    if displacement_dataset is not None:
        ph.dataset = displacement_dataset.get_dict()
    elif displacements is not None:
        ph.dataset = {"displacements": displacements.get_array("displacements")}
    else:
        supported_keys = (
            "distance",
            "is_plusminus",
            "is_diagonal",
            "number_of_snapshots",
            "random_seed",
        )
        kwargs = {key: ph_settings[key] for key in ph_settings if key in supported_keys}
        ph.generate_displacements(**kwargs)

    _update_structure_info(ph_settings, ph)
    structures_dict = _generate_phonopy_structures(ph)
    return_vals = {"phonon_setting_info": Dict(dict=ph_settings)}
    return_vals.update(structures_dict)
    if displacement_dataset is None and displacements is None:
        if "displacements" in ph.dataset:
            disp_array = ArrayData()
            disp_array.set_array("displacements", ph.dataset["displacements"])
            return_vals["displacements"] = disp_array
        else:
            return_vals["displacement_dataset"] = Dict(dict=ph.dataset)

    return return_vals


@calcfunction
def setup_phono3py_calculation(
    phonon_settings: Dict,
    structure: StructureData,
    symmetry_tolerance: Float,
    displacement_dataset: Optional[Dict] = None,
    displacements: Optional[ArrayData] = None,
    phonon_displacement_dataset: Optional[Dict] = None,
    phonon_displacements: Optional[ArrayData] = None,
):
    """Set up phono3py calculation.

    Returns
    -------
    dict
        'supercell' : StructureData
            Perfect supercell.
        'supercell_001', 'supercell_002', ... : StructureData
            Supercells with displacements
        'primitive' : StructureData
            Primitive cell.
        'phonon_supercell' : StructureData
            For phono3py. Perfect supercell for harmonic phonon calculation.
        'phonon_supercell_001', 'phonon_supercell_002', ... : StructureData
            For phono3py. Supercells with displacements for harmonic phonon
            calculation.
        'phonon_setting_info' : Dict
            Phonopy setting parameters including those generated in the
            process of displacements creation, e.g., primitive and  sueprcell
            matrix and symmetry information.

    phonon_setting_info contains the following entries:
        'supercell_matrix' : array_like
            3x3 integer matrix to generate supercell matrix.
        'phonon_supercell_matrix' : array_like
            3x3 integer matrix to generate fc2 supercell matrix for Phono3py.
        'distance' : float
            Displacement distance.
        'symmetry_tolerance' : float
            Tolerance length used for symmetry finding.
        'displacement_dataset' : dict
            Phonopy.dataset or Phono3py.dataset.
        'primitive_matrix' : array_like
            Phonopy.primitive_matrix.
        'symmetry' : dict
            'number' : Space group number.
            'international' : Space group type.
        'phonon_displacement_dataset' : dict, optional
            Phono3py.phonon_dataset.
        'random_seed' : int, optional (no support)
        'is_plusminus' : str or bool, optional
        'is_diagonal' : bool, optional

    """
    ph_settings = _get_setting_info(phonon_settings, code_name="phono3py")
    ph = _get_phono3py_instance(
        structure,
        ph_settings,
        symmetry_tolerance=symmetry_tolerance.value,
    )
    ph_settings["version"] = ph.version

    if displacement_dataset is not None:
        ph.dataset = displacement_dataset.get_dict()
    elif displacements is not None:
        ph.dataset = {"displacements": displacements.get_array("displacements")}
    if phonon_displacement_dataset is not None:
        ph.phonon_dataset = phonon_displacement_dataset.get_dict()
    elif phonon_displacements is not None:
        ph.phonon_dataset = {
            "displacements": phonon_displacements.get_array("displacements")
        }
    else:
        supported_keys = (
            "distance",
            "is_plusminus",
            "is_diagonal",
        )
        kwargs = {key: ph_settings[key] for key in ph_settings if key in supported_keys}
        ph.generate_displacements(**kwargs)

    _update_structure_info(ph_settings, ph)
    if "phonon_supercell_matrix" in ph_settings:
        if ph.phonon_supercell_matrix is not None:
            ph_settings["phonon_displacement_dataset"] = ph.phonon_dataset
    structures_dict = _generate_phonopy_structures(ph)
    if ph.phonon_supercell_matrix is not None:
        structures_dict.update(_generate_phono3py_phonon_structures(ph))
    return_vals = {"phonon_setting_info": Dict(dict=ph_settings)}
    return_vals.update(structures_dict)
    if displacement_dataset is None and displacements is None:
        if "displacements" in ph.dataset:
            disp_array = ArrayData()
            disp_array.set_array("displacements", ph.dataset["displacements"])
            return_vals["displacements"] = disp_array
        else:
            return_vals["displacement_dataset"] = Dict(dict=ph.dataset)
    if ph.phonon_dataset is not None:
        if phonon_displacement_dataset is None and phonon_displacements is None:
            if "displacements" in ph.phonon_dataset:
                disp_array = ArrayData()
                disp_array.set_array(
                    "displacements", ph.phonon_dataset["displacements"]
                )
                return_vals["phonon_displacements"] = disp_array
            else:
                return_vals["phonon_displacement_dataset"] = Dict(
                    dict=ph.phonon_dataset
                )

    return return_vals


def _get_phonopy_postprocess_info(phonon_settings: Dict) -> dict:
    """Return phonopy postprocess parameters."""
    valid_keys = ("mesh", "fc_calculator")
    params = {}
    for key in valid_keys:
        if key in phonon_settings.keys():
            params[key] = phonon_settings[key]

    if "mesh" not in phonon_settings.keys():
        params["mesh"] = 100.0
    return params


@calcfunction
def get_force_constants(
    structure: StructureData,
    phonon_setting_info: Dict,
    force_sets: ArrayData,
    symmetry_tolerance: Float,
    displacement_dataset: Optional[Dict] = None,
    displacements: Optional[ArrayData] = None,
):
    """Calculate force constants."""
    phonon = _get_phonopy_instance(
        structure, phonon_setting_info, symmetry_tolerance=symmetry_tolerance.value
    )
    if displacement_dataset is not None:
        dataset = displacement_dataset.get_dict()
    elif displacements is not None:
        dataset = {"displacements": displacements.get_array("displacements")}
    else:
        raise RuntimeError("Displacement dataset not found.")
    phonon.dataset = dataset
    phonon.forces = force_sets.get_array("force_sets")

    if "fc_calculator" in phonon_setting_info.keys():
        if phonon_setting_info["fc_calculator"].lower().strip() == "alm":
            phonon.produce_force_constants(fc_calculator="alm")
    else:
        phonon.produce_force_constants()
    force_constants = ArrayData()
    force_constants.set_array("force_constants", phonon.force_constants)
    force_constants.set_array("p2s_map", phonon.primitive.p2s_map)
    force_constants.label = "force_constants"

    return force_constants


@calcfunction
def get_phonon_properties(
    structure: StructureData,
    phonon_setting_info: Dict,
    force_constants: ArrayData,
    symmetry_tolerance: Float,
    nac_params: Optional[ArrayData] = None,
):
    """Calculate phonon properties."""
    phonon_settings_dict = phonon_setting_info.get_dict()
    ph = _get_phonopy_instance(
        structure,
        phonon_settings_dict,
        symmetry_tolerance=symmetry_tolerance.value,
        nac_params=nac_params,
    )
    ph.force_constants = force_constants.get_array("force_constants")
    mesh = phonon_settings_dict.get("mesh", None)

    # Mesh
    total_dos, pdos, thermal_properties = get_mesh_property_data(ph, mesh)

    # Band structure
    bs = _get_bands_data(ph)

    return {
        "dos": total_dos,
        "pdos": pdos,
        "thermal_properties": thermal_properties,
        "band_structure": bs,
    }


def compare_structures(structure_a, structure_b, symprec=1e-5):
    """Compare two structures."""
    cell_a = structure_a.cell
    cell_b = structure_b.cell
    cell_diff = np.subtract(cell_a, cell_b)
    if (np.abs(cell_diff) > symprec).any():
        return False

    for site_a, site_b in zip(structure_a.sites, structure_b.sites):
        if site_a.kind_name != site_b.kind_name:
            return False

    positions_a = [site.position for site in structure_a.sites]
    frac_positions_a = np.dot(positions_a, np.linalg.inv(cell_a))
    positions_b = [site.position for site in structure_b.sites]
    frac_positions_b = np.dot(positions_b, np.linalg.inv(cell_b))
    diff = frac_positions_a - frac_positions_b
    diff -= np.rint(diff)
    dist = np.sqrt(np.sum(np.dot(diff, structure_a.cell) ** 2, axis=1))
    if (dist > symprec).any():
        return False

    return True


def get_structure_from_vasp_immigrant(wc_node):
    """Get structure from VASP immigrant workchain.

    VaspImmigrantWorkChain doesn't have inputs.structure but
    VaspCalculation does. VaspImmigrantWorkChain is a sub-class of
    RestartWorkChain, so VaspCalculation has a link_label of
    "iteration_{num}". Here, no failure of VaspCalculation is assumed.

    """
    for lt in wc_node.get_outgoing():
        if "iteration_" in lt.link_label:
            structure = lt.node.inputs.structure
            return structure
    return None


def get_mesh_property_data(ph, mesh):
    """Return total DOS, PDOS, thermal properties."""
    ph.set_mesh(mesh)
    ph.run_total_dos()

    dos = get_total_dos(ph.get_total_dos_dict())

    ph.run_thermal_properties()
    tprops = get_thermal_properties(ph.get_thermal_properties_dict())

    ph.set_mesh(mesh, is_eigenvectors=True, is_mesh_symmetry=False)
    ph.run_projected_dos()
    pdos = get_projected_dos(ph.get_projected_dos_dict())

    return dos, pdos, tprops


def get_total_dos(total_dos):
    """Return XyData of total DOS."""
    dos = XyData()
    dos.set_x(total_dos["frequency_points"], "Frequency", "THz")
    dos.set_y(total_dos["total_dos"], "Total DOS", "1/THz")
    dos.label = "Total DOS"
    return dos


def get_projected_dos(projected_dos):
    """Return XyData of PDOS."""
    pdos = XyData()
    pdos_list = [pd for pd in projected_dos["projected_dos"]]
    pdos.set_x(projected_dos["frequency_points"], "Frequency", "THz")
    pdos.set_y(
        pdos_list,
        [
            "Projected DOS",
        ]
        * len(pdos_list),
        [
            "1/THz",
        ]
        * len(pdos_list),
    )
    pdos.label = "Projected DOS"
    return pdos


def get_thermal_properties(thermal_properties):
    """Return XyData of thermal properties."""
    tprops = XyData()
    tprops.set_x(thermal_properties["temperatures"], "Temperature", "K")
    tprops.set_y(
        [
            thermal_properties["free_energy"],
            thermal_properties["entropy"],
            thermal_properties["heat_capacity"],
        ],
        ["Helmholtz free energy", "Entropy", "Cv"],
        ["kJ/mol", "J/K/mol", "J/K/mol"],
    )
    tprops.label = "Thermal properties"
    return tprops


def _get_bands_data(ph):
    ph.auto_band_structure()
    labels = [
        x.replace("$", "")
        .replace("\\", "")
        .replace("mathrm{", "")
        .replace("}", "")
        .upper()
        for x in ph.band_structure.labels
    ]
    frequencies = ph.band_structure.frequencies
    qpoints = ph.band_structure.qpoints
    path_connections = ph.band_structure.path_connections
    label = "%s (%d)" % (
        ph.symmetry.dataset["international"],
        ph.symmetry.dataset["number"],
    )

    return get_bands(qpoints, frequencies, labels, path_connections, label=label)


def get_bands(qpoints, frequencies, labels, path_connections, label=None):
    """Return BandsData."""
    qpoints_list = list(qpoints[0])
    frequencies_list = list(frequencies[0])
    labels_list = [
        (0, labels[0]),
    ]
    label_index = 1

    for pc, qs, fs in zip(path_connections[:-1], qpoints[1:], frequencies[1:]):
        if labels[label_index] == "GAMMA" and pc:
            labels_list.append((len(qpoints_list) - 1, labels[label_index]))
            if label_index < len(labels):
                labels_list.append((len(qpoints_list), labels[label_index]))
            label_index += 1
            qpoints_list += list(qs)
            frequencies_list += list(fs)
        elif pc:
            labels_list.append((len(qpoints_list) - 1, labels[label_index]))
            label_index += 1
            qpoints_list += list(qs[1:])
            frequencies_list += list(fs[1:])
        else:
            labels_list.append((len(qpoints_list) - 1, labels[label_index]))
            label_index += 1
            if label_index < len(labels):
                labels_list.append((len(qpoints_list), labels[label_index]))
                label_index += 1
            qpoints_list += list(qs)
            frequencies_list += list(fs)
    labels_list.append((len(qpoints_list) - 1, labels[-1]))

    bs = BandsData()
    bs.set_kpoints(np.array(qpoints_list))
    bs.set_bands(np.array(frequencies_list), units="THz")
    bs.labels = labels_list
    if label is not None:
        bs.label = label

    return bs


def get_kpoints_data(kpoints_dict, structure=None):
    """Return KpointsData from arguments.

    kpoints_dict : dict
        Supported keys:
            "kpoints_density" (structure required)
            "kpoints_mesh"
            "kpoints_offset"
    structure : StructureData
        A structure.

    """
    kpoints = KpointsData()
    if "kpoints_density" in kpoints_dict.keys():
        kpoints.set_cell_from_structure(structure)
        kpoints.set_kpoints_mesh_from_density(kpoints_dict["kpoints_density"])
    elif "kpoints_mesh" in kpoints_dict.keys():
        if "kpoints_offset" in kpoints_dict.keys():
            kpoints_offset = kpoints_dict["kpoints_offset"]
        else:
            kpoints_offset = [0.0, 0.0, 0.0]

        kpoints.set_kpoints_mesh(kpoints_dict["kpoints_mesh"], offset=kpoints_offset)
    else:
        raise InputValidationError(
            "no kpoint definition in input. "
            "Define either kpoints_density or kpoints_mesh"
        )
    return kpoints


def _get_phonopy_instance(
    structure: StructureData,
    phonon_settings_dict: dict,
    nac_params: Optional[ArrayData] = None,
    symmetry_tolerance: float = 1e-5,
) -> Phonopy:
    """Create Phonopy instance."""
    phpy = Phonopy(
        phonopy_atoms_from_structure(structure),
        supercell_matrix=phonon_settings_dict["supercell_matrix"],
        primitive_matrix="auto",
        symprec=symmetry_tolerance,
    )
    if nac_params:
        _set_nac_params(phpy, nac_params)
    return phpy


def _get_phono3py_instance(
    structure: StructureData,
    phonon_settings_dict: dict,
    nac_params: Optional[ArrayData] = None,
    symmetry_tolerance: float = 1e-5,
):
    """Create Phono3py instance."""
    from phono3py import Phono3py

    if "phonon_supercell_matrix" in phonon_settings_dict:
        ph_smat = phonon_settings_dict["phonon_supercell_matrix"]
    else:
        ph_smat = None
    ph3py = Phono3py(
        phonopy_atoms_from_structure(structure),
        supercell_matrix=phonon_settings_dict["supercell_matrix"],
        primitive_matrix="auto",
        phonon_supercell_matrix=ph_smat,
        symprec=symmetry_tolerance,
    )
    if nac_params:
        _set_nac_params(ph3py, nac_params["nac_params"])
    return ph3py


def _set_nac_params(phpy: Phonopy, nac_params: ArrayData) -> None:
    units = get_default_physical_units("vasp")
    factor = units["nac_factor"]
    nac_params = {
        "born": nac_params.get_array("born_charges"),
        "dielectric": nac_params.get_array("epsilon"),
        "factor": factor,
    }
    phpy.nac_params = nac_params


def phonopy_atoms_to_structure(cell):
    """Convert PhonopyAtoms to StructureData."""
    symbols = cell.symbols
    positions = cell.positions
    structure = StructureData(cell=cell.cell)
    for symbol, position in zip(symbols, positions):
        structure.append_atom(position=position, symbols=symbol)
    return structure


def phonopy_atoms_from_structure(structure):
    """Convert StructureData to PhonopyAtoms."""
    cell = PhonopyAtoms(
        symbols=[site.kind_name for site in structure.sites],
        positions=[site.position for site in structure.sites],
        cell=structure.cell,
    )
    return cell


def collect_forces_and_energies(ctx, ctx_supercells, prefix="force_calc"):
    """Collect forces and energies from calculation outputs.

    Parameters
    ----------
    ctx : AttributeDict-like
        AiiDA workchain context.
    ctx_supercells : dict of StructDict
        Supercells. For phono3py, this can be phonon_supercells.
    prefix : str
        Prefix string of dictionary keys of ctx.

    Returns
    -------
    dict
        Forces and energies.

    """
    forces_dict = {}
    for key in ctx_supercells:
        # key: e.g. "supercell_001", "phonon_supercell_001"
        num = key.split("_")[-1]  # e.g. "001"
        calc = ctx["%s_%s" % (prefix, num)]
        if type(calc) is dict:
            calc_dict = calc
        else:
            calc_dict = calc.outputs
        forces_dict["forces_%s" % num] = calc_dict["forces"]
        if "energy" in calc_dict:
            forces_dict["energy_%s" % num] = calc_dict["energy"]
        elif "total_energy" in calc_dict:  # For CommonWorkflow
            forces_dict["energy_%s" % num] = calc_dict["total_energy"]

    return forces_dict


@calcfunction
def get_force_sets(**forces_dict):
    """Create force sets from supercell forces.

    Parameters
    ----------
    forces_dict : dict
        'forces_001', 'forces_002', ... have to exist.
        'energy_001', 'energy_002', ... are optional.
        'forces_000' and 'energy_000' for perfect supercell are optional.
        The zero-padding length of the numbers can change depending on total
        number of supercell calculations.

    """
    (force_sets, energies, forces_0_key, energy_0_key) = _get_force_set(**forces_dict)

    force_sets_data = ArrayData()
    force_sets_data.set_array("force_sets", force_sets)
    if energies is not None:
        force_sets_data.set_array("energies", energies)
    force_sets_data.label = "force_sets"
    ret_dict = {"force_sets": force_sets_data}
    if forces_0_key is not None:
        ret_dict["supercell_forces"] = forces_dict[forces_0_key]
    if energy_0_key is not None:
        ret_dict["supercell_energy"] = forces_dict[energy_0_key]

    return ret_dict


def _get_force_set(**forces_dict):
    num_forces = 0
    num_energies = 0
    forces_0_key = None
    energy_0_key = None
    shape = None
    for key in forces_dict:
        value = forces_dict[key]
        if int(key.split("_")[-1]) != 0:
            if "forces" in key:
                num_forces += 1
                if shape is None:
                    shape = value.get_array("forces").shape
            elif "energy" in key:
                num_energies += 1
        else:
            if "forces" in key:
                forces_0_key = key
            elif "energy" in key:
                energy_0_key = key

    force_sets = np.zeros((num_forces,) + shape, dtype=float)
    if num_energies > 0:
        energies = np.zeros(num_energies, dtype=float)
    else:
        energies = None
    if forces_0_key is None:
        forces_0 = None
    else:
        forces_0 = forces_dict[forces_0_key].get_array("forces")

    for key in forces_dict:
        value = forces_dict[key]
        num = int(key.split("_")[-1])  # e.g. "001" --> 1
        if "forces" in key:
            forces = value.get_array("forces")
            if forces_0 is None:
                force_sets[num - 1] = forces
            else:
                force_sets[num - 1] = forces - forces_0
        elif "energy" in key:
            if isinstance(value, Float):  # For CommonWorkflow
                energies[num - 1] = value.value
            else:
                energies[num - 1] = value.get_array("energy")[-1]

    return force_sets, energies, forces_0_key, energy_0_key


def _get_setting_info(phonon_settings: Dict, code_name: str = "phonopy") -> dict:
    """Convert AiiDA inputs to a dict.

    code_name : 'phonopy' or 'phono3py'

    Note
    ----
    Designed to be shared by phonopy and phono3py.

    Returns
    -------
    dict
        'supercell_matrix' : ndarray
            3x3 integer matrix to generate supercell matrix.
        'phonon_supercell_matrix' : ndarray
            3x3 integer matrix to generate fc2 supercell matrix for Phono3py.
        'mesh' : list or float
            Mesh numbers or a distance to represent mesh numbers.
        'distance' : float
            Displacement distance.
        'symmetry_tolerance' : float
            Tolerance length used for symmetry finding.
        'random_seed' : int, optional
        'is_plusminus' : bool, optional
        'is_diagonal' : bool, optional

    """
    ph_settings = {}
    valid_keys = (
        "supercell_matrix",
        "phonon_supercell_matrix",
        "distance",
        "symmetry_tolerance",
        "number_of_snapshots",
        "random_seed",
        "is_plusminus",
        "is_diagonal",
    )
    for key, value in phonon_settings.get_dict().items():
        if key in valid_keys:
            ph_settings[key] = value
    dim = ph_settings["supercell_matrix"]
    ph_settings["supercell_matrix"] = _get_supercell_matrix(dim)
    if "code_name" == "phono3py" and "phonon_supercell_matrix" in ph_settings:
        ph_settings["phonon_supercell_matrix"] = _get_supercell_matrix(
            ph_settings["phonon_supercell_matrix"], smat_type="phonon_supercell_matrix"
        )
    if "distance" not in ph_settings:
        _set_displacement_distance(ph_settings, code_name)

    return ph_settings


def _set_displacement_distance(ph_settings, code_name):
    if code_name == "phono3py":
        from phono3py.interface.calculator import get_default_displacement_distance
    else:
        from phonopy.interface.calculator import get_default_displacement_distance
    distance = get_default_displacement_distance("vasp")
    ph_settings["distance"] = distance


def _get_supercell_matrix(dim, smat_type="supercell_matrix"):
    if len(np.ravel(dim)) == 3:
        smat = np.diag(dim)
    else:
        smat = np.array(dim)
    if not np.issubdtype(smat.dtype, np.integer):
        raise TypeError("%s is not integer matrix." % smat_type)
    else:
        return smat.tolist()


def _update_structure_info(ph_settings, ph):
    """Update a phonon_settings dict.

    Parameters
    ----------
    ph_settings : Dict
         Phonopy setting information.
    ph : Phonopy or Phono3py
         A Phonopy or Phono3py instance.

    Returns
    -------
    dict
        'primitive_matrix' : ndarray
            Phonopy.primitive_matrix.
        'symmetry' : dict
            'number' : Space group number.
            'international' : Space group type.

    """
    ph_settings["primitive_matrix"] = ph.primitive_matrix
    ph_settings["symmetry"] = {
        "number": ph.symmetry.dataset["number"],
        "international": ph.symmetry.dataset["international"],
    }

    return ph_settings


def _generate_phonopy_structures(ph):
    """Generate AiiDA structures of phonon related cells.

    Note
    ----
    Designed to be shared by phonopy and phono3py.
    ph is either an instance of Phonopy or Phono3py.

    Returns
    -------
    dict of StructureData
        'supercell'
            Perfect supercell.
        'supercell_001', 'supercell_002', ...
            Supercells with displacements
        'primitive':
            Primitive cell.
        'phonon_supercell'
            For phono3py. Perfect supercell for harmonic phonon calculation.
        'phonon_supercell_001', 'phonon_supercell_002', ...
            For phono3py. Supercells with displacements for harmonic phonon
            calculation.

    """
    structures_dict = _generate_supercell_structures(
        ph.supercell, ph.supercells_with_displacements
    )
    primitive_structure = phonopy_atoms_to_structure(ph.primitive)
    formula = primitive_structure.get_formula(mode="hill_compact")
    primitive_structure.label = f"{formula} primitive cell"
    structures_dict["primitive"] = primitive_structure
    return structures_dict


def _generate_phono3py_phonon_structures(ph):
    """Generate AiiDA structures of phono3py phonon related cells.

    Returns
    -------
    dict of StructureData
        'phonon_supercell'
            For phono3py. Perfect supercell for harmonic phonon calculation.
        'phonon_supercell_001', 'phonon_supercell_002', ...
            For phono3py. Supercells with displacements for harmonic phonon
            calculation.

    """
    return _generate_supercell_structures(
        ph.phonon_supercell,
        ph.phonon_supercells_with_displacements,
        label_prefix="phonon_supercell",
    )


def _generate_supercell_structures(
    supercell, supercells_with_displacements, label_prefix="supercell"
):
    """Generate AiiDA supercell structures.

    Note
    ----
    Designed to be shared by phonopy and phono3py.
    ph is either an instance of Phonopy or Phono3py.

    Returns
    -------
    Possible dict keys
        'supercell'
            Perfect supercell.
        'supercell_001', 'supercell_002', ...
            Supercells with displacements
        'phonon_supercell'
            For phono3py. Perfect supercell for harmonic phonon calculation.
        'phonon_supercell_001', 'phonon_supercell_002', ...
            For phono3py. Supercells with displacements for harmonic phonon
            calculation.

    """
    structures_dict = {}
    supercell_structure = phonopy_atoms_to_structure(supercell)
    formula = supercell_structure.get_formula(mode="hill_compact")
    supercell_structure.label = f"{formula} {label_prefix}"
    structures_dict[label_prefix] = supercell_structure

    digits = len(str(len(supercells_with_displacements)))
    for i, scell in enumerate(supercells_with_displacements):
        structure = phonopy_atoms_to_structure(scell)
        num = str(i + 1).zfill(digits)
        label = f"{label_prefix}_{num}"
        structure.label = f"{formula} {label}"
        structures_dict[label] = structure

    return structures_dict
