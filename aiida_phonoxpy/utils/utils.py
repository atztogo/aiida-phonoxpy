"""General utilities."""

from typing import Optional

import numpy as np
from aiida.common import InputValidationError
from aiida.engine import calcfunction
from aiida.orm import (
    ArrayData,
    BandsData,
    Bool,
    Dict,
    Float,
    KpointsData,
    StructureData,
    XyData,
)
from phonopy import Phonopy
from phonopy.interface.calculator import get_default_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.dataset import get_displacements_and_forces


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

    Default valide keys in 'phonon_settings_info` are (1)
    ```
        ('supercell_matrix',
         'primitive_matrix',
         'symmetry',
         'symmetry_tolerance',
         'version')
    ```
    and, when `run_phonopy=True`, (2)
    ```
        ('mesh', 'fc_calculator')
    ```
    and, when displacements are generated, (3)
    ```
        ('distance',
         'number_of_snapshots',
         'random_seed',
         'is_plusminus',
         'is_diagonal')
    ```.

    `primitive_matrix` is always `auto` and the determined `primitive_matrix` by
    phonopy is stored in returned `phonon_setting_info`.

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
            3x3 integer matrix to generate supercell.
        'primitive_matrix': array_like
            3x3 float matrix to generate primitive cell.
        'symmetry_tolerance' : float
            Tolerance length used for symmetry finding.
        'symmetry' : dict
            'number' : Space group number.
            'international' : Space group type.
        'distance' : float, optional
            Displacement distance.
        'number_of_snapshots' : int, optional
        'random_seed' : int, optional
        'is_plusminus' : str or bool, optional
        'is_diagonal' : bool, optional
        'mesh' : float or list, optional
            Mesh numbers or distance measure of q-point sampling mesh.
        'fc_calculator' : str, optional
            External force constants calculator.

    """
    # Key-set 1
    ph_settings = {"symmetry_tolerance": symmetry_tolerance.value}
    ph_settings["supercell_matrix"] = _get_supercell_matrix(
        phonon_settings["supercell_matrix"]
    )
    ph = get_phonopy_instance(structure, ph_settings)
    ph_settings["primitive_matrix"] = ph.primitive_matrix
    ph_settings["version"] = ph.version
    _set_symmetry_info(ph_settings, ph)

    # Key-set 2
    if run_phonopy:
        valid_keys = ("mesh", "fc_calculator")
        for key in valid_keys:
            if key in phonon_settings.keys():
                ph_settings[key] = phonon_settings[key]
        if "mesh" not in ph_settings:
            ph_settings["mesh"] = 100.0

    if displacement_dataset is not None:
        ph.dataset = displacement_dataset.get_dict()
    elif displacements is not None:
        ph.dataset = {"displacements": displacements.get_array("displacements")}
    else:
        # Key-set 3
        ph_settings["distance"] = _get_default_displacement_distance("phonopy")
        supported_keys = (
            "distance",
            "is_plusminus",
            "is_diagonal",
            "number_of_snapshots",
            "random_seed",
        )
        kwargs = {
            key: phonon_settings[key]
            for key in phonon_settings.keys()
            if key in supported_keys
        }
        ph_settings.update(kwargs)
        ph.generate_displacements(**kwargs)

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
    run_fc: Bool,
    run_ltc: Bool,
    displacement_dataset: Optional[Dict] = None,
    displacements: Optional[ArrayData] = None,
    phonon_displacement_dataset: Optional[Dict] = None,
    phonon_displacements: Optional[ArrayData] = None,
):
    """Set up phono3py calculation.

    Key sets in 'phonon_settings_info` are:
    (1) Minimum set for forces and force constants calculation.
    ```
        ('supercell_matrix', 'phonon_supercell_matrix', 'symmetry_tolerance', 'version')
    ```
    (2) Additional information about geometry.
    ```
        ('primitive_matrix', 'symmetry')
    ```
    (3) Detailed controling parameters to constructure displacements.
    ```
        ('distance',
         'number_of_snapshots',
         'phonon_number_of_snapshots',
         'random_seed',
         'is_plusminus',
         'is_diagonal')
    ```
    (4) Force constants (`run_fc=True`), LTC calculation (`run_ltc=True`).
    ```
        ('fc_calculator', 'mesh', 'isotope', 'br', 'lbte', 'ts', 'grg')
    ```

    `primitive_matrix` is always `auto` and the determined `primitive_matrix` by
    phonopy is stored in returned `phonon_setting_info`.

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
        'phonon_displacement_dataset' : Dict, optional
            When 'phonon_number_of_snapshots' is not given, displacements for fc2 are
            generated in a usual way (least number of displacements considering
            symmetry), and phonopy's type-I displacement dataset is returned.
        'phonon_displacements' : ArrayData, optional
            When 'phonon_number_of_snapshots' is given, random displacements are
            generated for fc2 and phonopy's type-II displacements array is returned.

    phonon_setting_info contains the following entries:
        'version' : str
            Phono3py version number.
        'supercell_matrix' : array_like
            3x3 integer matrix to generate supercell matrix.
        'phonon_supercell_matrix' : array_like
            3x3 integer matrix to generate fc2 supercell matrix for Phono3py.
        'primitive_matrix': array_like
            3x3 float matrix to generate primitive cell.
        'symmetry_tolerance' : float
            Tolerance length used for symmetry finding.
        'symmetry' : dict
            'number' : Space group number.
            'international' : Space group type.
        'distance' : float, optional
            Displacement distance.
        'number_of_snapshots' : int, optional
        'phonon_number_of_snapshots' : int, optional
        'random_seed' : int, optional
        'is_plusminus' : str or bool, optional
        'is_diagonal' : bool, optional

    """
    ph_settings: dict = {}
    return_vals = {}
    structures_dict = {}

    # Key-set 1
    _setup_phono3py_calculation_keyset1(
        ph_settings, phonon_settings, structure, symmetry_tolerance.value
    )

    # Key-set 2
    ph = get_phono3py_instance(structure, ph_settings)
    _set_symmetry_info(ph_settings, ph)

    # Key-set 3
    if displacement_dataset is not None:
        ph.dataset = displacement_dataset.get_dict()
    elif displacements is not None:
        ph.dataset = {"displacements": displacements.get_array("displacements")}
    else:
        ph_settings["distance"] = _get_default_displacement_distance("phono3py")
        supported_keys = (
            "distance",
            "is_plusminus",
            "is_diagonal",
        )
        kwargs = {
            key: phonon_settings[key]
            for key in phonon_settings.keys()
            if key in supported_keys
        }
        ph_settings.update(kwargs)
        ph.generate_displacements(**kwargs)
        structures_dict.update(_generate_phonopy_structures(ph))

    if displacement_dataset is None and displacements is None:
        if "displacements" in ph.dataset:
            disp_array = ArrayData()
            disp_array.set_array("displacements", ph.dataset["displacements"])
            return_vals["displacements"] = disp_array
        else:
            return_vals["displacement_dataset"] = Dict(dict=ph.dataset)

    if "phonon_supercell_matrix" in ph_settings:
        if phonon_displacement_dataset is not None:
            ph.phonon_dataset = phonon_displacement_dataset.get_dict()
        elif phonon_displacements is not None:
            ph.phonon_dataset = {
                "displacements": phonon_displacements.get_array("displacements")
            }
        elif displacement_dataset is not None or displacements is not None:
            # datasets for supercell_matrix is supplied in wc inputs. Therefore
            # parameters for generate displacements are not yet stored in
            # ph_settings. Here, these parameters are stored because they are used
            # for phonon_supercells.
            ph_settings["distance"] = _get_default_displacement_distance("phono3py")
            supported_keys = (
                "distance",
                "is_plusminus",
                "is_diagonal",
            )
            kwargs = {
                key: phonon_settings[key]
                for key in phonon_settings.keys()
                if key in supported_keys
            }
            ph_settings.update(kwargs)
            ph.generate_fc2_displacements(**kwargs)

        structures_dict.update(_generate_phono3py_phonon_structures(ph))
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

    # Key-set 4
    _setup_phono3py_calculation_keyset4(
        ph_settings, phonon_settings, run_fc=run_fc, run_ltc=run_ltc
    )

    if structures_dict:
        return_vals.update(structures_dict)
    return_vals["phonon_setting_info"] = Dict(dict=ph_settings)

    return return_vals


def _setup_phono3py_calculation_keyset1(
    ph_settings: dict, phonon_settings: Dict, structure: StructureData, symprec: float
):
    import phono3py

    ph_settings["symmetry_tolerance"] = symprec
    ph_settings["supercell_matrix"] = _get_supercell_matrix(
        phonon_settings["supercell_matrix"]
    )
    if "phonon_supercell_matrix" in phonon_settings.keys():
        ph_settings["phonon_supercell_matrix"] = _get_supercell_matrix(
            phonon_settings["phonon_supercell_matrix"],
            smat_type="phonon_supercell_matrix",
        )
    ph_settings["version"] = phono3py.__version__


def _setup_phono3py_calculation_keyset4(
    ph_settings: dict,
    phonon_settings: Dict,
    run_fc: bool = False,
    run_ltc: bool = False,
):
    """Set calculation options.

    Force constants calculation
    ---------------------------
    fc_calculator : str
        External force constants calculator.

    LTC calculation
    ---------------
    mesh : float, list
        Uniform sampling mesh.
    isotope : bool
        With / without isotope scattering
    br : bool
        Use RTA or not. This is the default behaviour.
    lbte : bool
        Use direct solution or not.
    ts : list
        Temperatures. The default value is [300].
    grg : bool
        Use generalized-regular grid or not.

    """
    if run_fc:
        for key in ("fc_calculator",):
            if key in phonon_settings.keys():
                ph_settings[key] = phonon_settings[key]

    if run_ltc:
        for key in ("mesh", "isotope", "lbte", "br", "ts", "grg"):
            if key in phonon_settings.keys():
                ph_settings[key] = phonon_settings[key]


@calcfunction
def setup_phono3py_fc_calculation(
    phonon_settings: Dict,
    structure: StructureData,
    symmetry_tolerance: Float,
):
    """Set up phono3py force constants calculation.

    Returns
    -------
    key-set 1 as described in `setup_phono3py_calculation`.

    """
    ph_settings: dict = {}
    return_vals = {}

    # Key-set 1
    _setup_phono3py_calculation_keyset1(
        ph_settings, phonon_settings, structure, symmetry_tolerance.value
    )

    # Key-set 4
    _setup_phono3py_calculation_keyset4(ph_settings, phonon_settings, run_fc=True)

    return_vals["phonon_setting_info"] = Dict(dict=ph_settings)
    return return_vals


@calcfunction
def setup_phono3py_ltc_calculation(
    phonon_settings: Dict,
    structure: StructureData,
    symmetry_tolerance: Float,
) -> dict:
    """Set up phono3py lattice thermal conductivity calculation.

    Returns
    -------
    * key-set 1 amd 2 as described in `setup_phono3py_calculation`.

    """
    ph_settings: dict = {}
    return_vals = {}

    # Key-set 1
    _setup_phono3py_calculation_keyset1(
        ph_settings, phonon_settings, structure, symmetry_tolerance.value
    )

    # Key-set 2
    ph = get_phono3py_instance(structure, ph_settings)
    _set_symmetry_info(ph_settings, ph)

    # Key-set 4
    _setup_phono3py_calculation_keyset4(ph_settings, phonon_settings, run_ltc=True)

    return_vals["phonon_setting_info"] = Dict(dict=ph_settings)
    return return_vals


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
    phonon = get_phonopy_instance(structure, phonon_setting_info.get_dict())
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
    nac_params: Optional[ArrayData] = None,
):
    """Calculate phonon properties."""
    phonon_settings_dict = phonon_setting_info.get_dict()
    ph = get_phonopy_instance(
        structure,
        phonon_settings_dict,
        nac_params=nac_params,
    )
    ph.force_constants = force_constants.get_array("force_constants")
    mesh = phonon_settings_dict.get("mesh", None)

    # Mesh
    total_dos, pdos, thermal_properties = get_mesh_property_data(ph, mesh)

    # Band structure
    bs = _get_bands_data(ph)

    return {
        "total_dos": total_dos,
        "projected_dos": pdos,
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
    ph.run_mesh(mesh=mesh)
    ph.run_total_dos()

    dos = get_total_dos(ph.get_total_dos_dict())

    ph.run_thermal_properties()
    tprops = get_thermal_properties(ph.get_thermal_properties_dict())

    ph.run_mesh(mesh=mesh, with_eigenvectors=True, is_mesh_symmetry=False)
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


def get_phonopy_instance(
    structure: StructureData,
    phonon_settings_dict: dict,
    nac_params: Optional[ArrayData] = None,
) -> Phonopy:
    """Create Phonopy instance."""
    if "primitive_matrix" in phonon_settings_dict:
        primitive_matrix = phonon_settings_dict["primitive_matrix"]
    else:
        primitive_matrix = "auto"
    kwargs = {
        "supercell_matrix": phonon_settings_dict["supercell_matrix"],
        "primitive_matrix": primitive_matrix,
    }
    if "symmetry_tolerance" in phonon_settings_dict:
        kwargs["symprec"] = phonon_settings_dict["symmetry_tolerance"]
    phpy = Phonopy(phonopy_atoms_from_structure(structure), **kwargs)
    if nac_params:
        _set_nac_params(phpy, nac_params)
    return phpy


def get_phono3py_instance(
    structure: StructureData,
    phonon_settings_dict: dict,
    nac_params: Optional[ArrayData] = None,
):
    """Create Phono3py instance."""
    from phono3py import Phono3py

    kwargs = {"supercell_matrix": phonon_settings_dict["supercell_matrix"]}
    if "symmetry_tolerance" in phonon_settings_dict:
        kwargs["symprec"] = phonon_settings_dict["symmetry_tolerance"]
    else:
        kwargs["symprec"] = 1e-5
    if "primitive_matrix" in phonon_settings_dict:
        kwargs["primitive_matrix"] = phonon_settings_dict["primitive_matrix"]
    else:
        kwargs["primitive_matrix"] = "auto"
    if "phonon_supercell_matrix" in phonon_settings_dict:
        kwargs["phonon_supercell_matrix"] = phonon_settings_dict[
            "phonon_supercell_matrix"
        ]
    ph3py = Phono3py(phonopy_atoms_from_structure(structure), **kwargs)
    if nac_params:
        _set_nac_params(ph3py, nac_params)

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


def collect_forces_and_energies(ctx, ctx_supercells, calc_key_prefix="force_calc"):
    """Collect forces and energies from calculation outputs.

    Parameters
    ----------
    ctx : AttributeDict-like
        AiiDA workchain context.
    ctx_supercells : dict of StructDict
        Supercells. For phono3py, this can be phonon_supercells.
    calc_key_prefix : str
        Prefix string of dictionary keys of calculation processes in ctx.

    Returns
    -------
    dict
        Forces and energies.

    """
    forces_dict = {}
    for key in ctx_supercells:
        # key: e.g. "supercell_001", "phonon_supercell_001"
        num = key.split("_")[-1]  # e.g. "001"
        calc = ctx[f"{calc_key_prefix}_{num}"]
        if isinstance(calc, dict):
            calc_dict = calc
        else:
            calc_dict = calc.outputs
        forces_dict[f"forces_{num}"] = calc_dict["forces"]
        if "energy" in calc_dict:
            forces_dict[f"energy_{num}"] = calc_dict["energy"]
        elif "total_energy" in calc_dict:  # For CommonWorkflow
            forces_dict[f"energy_{num}"] = calc_dict["total_energy"]

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
    force_sets, energies = _get_force_set(**forces_dict)

    force_sets_data = ArrayData()
    force_sets_data.set_array("force_sets", force_sets)
    if energies is not None:
        force_sets_data.set_array("energies", energies)
    force_sets_data.label = "force_sets"
    ret_dict = {"force_sets": force_sets_data}

    return ret_dict


def _get_force_set(**forces_dict):
    num_forces = 0
    num_energies = 0
    forces_0_key = None
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
        if num == 0:
            continue

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

    return force_sets, energies


def _get_default_displacement_distance(code_name: str) -> float:
    if code_name == "phono3py":
        from phono3py.interface.calculator import get_default_displacement_distance
    else:
        from phonopy.interface.calculator import get_default_displacement_distance
    distance = get_default_displacement_distance("vasp")
    return distance


def _get_supercell_matrix(dim, smat_type="supercell_matrix"):
    if len(np.ravel(dim)) == 3:
        smat = np.diag(dim)
    else:
        smat = np.array(dim)
    if not np.issubdtype(smat.dtype, np.integer):
        raise TypeError("%s is not integer matrix." % smat_type)
    else:
        return smat.tolist()


def _set_symmetry_info(ph_settings: dict, ph) -> None:
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


def _generate_phonopy_structures(ph) -> dict:
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


def get_displacements_from_phonopy_wc(node):
    """Return displacements ArrayData from output node of PhonopyWorkChain."""
    if "displacements" in node.outputs:
        return node.outputs.displacements

    if "displacement_dataset" in node.outputs:
        dataset = node.outputs.displacement_dataset.get_dict()
        d = ArrayData()
        d.set_array(
            "displacements",
            np.array(get_displacements_and_forces(dataset)[0], dtype="double"),
        )
        return d

    if "displacements" in node.inputs:
        return node.inputs.displacements

    if "displacement_dataset" in node.inputs:
        dataset = node.inputs.displacement_dataset.get_dict()
        d = ArrayData()
        d.set_array(
            "displacements",
            np.array(get_displacements_and_forces(dataset)[0], dtype="double"),
        )
        return d

    raise RuntimeError("displacements not found.")
