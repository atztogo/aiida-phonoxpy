"""Test phonopy parser."""
from aiida_phonoxpy.utils.utils import (
    phonopy_atoms_to_structure,
    phonopy_atoms_from_structure,
    compare_structures,
)
from phonopy.structure.cells import isclose


def test_phonopy_atoms_from_to_structure(ph_nacl):
    """Test phonopy_atoms_to/from_structure."""
    cell = ph_nacl.unitcell
    structure = phonopy_atoms_to_structure(cell)
    cell2 = phonopy_atoms_from_structure(structure)
    structure2 = phonopy_atoms_to_structure(cell2)
    assert isclose(cell, cell2)
    assert compare_structures(structure, structure2)


def test_set_masses_to_structure(ph_nacl):
    """Test structure set masses."""
    cell = ph_nacl.unitcell
    cell.masses = [
        30.0,
    ] * len(cell)
    structure = phonopy_atoms_to_structure(cell)
    cell2 = phonopy_atoms_from_structure(structure)
    structure2 = phonopy_atoms_to_structure(cell2)
    assert isclose(cell, cell2)
    assert compare_structures(structure, structure2)
