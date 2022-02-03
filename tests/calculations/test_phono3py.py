"""Test phono3py parser."""
import pytest
from aiida.common import AttributeDict
from aiida_phonoxpy.utils.utils import _setup_phono3py_calculation_keyset4


@pytest.fixture
def generate_inputs(generate_structure, generate_nac_params):
    """Return only those inputs that the parser will expect to be there."""

    def _generate_inputs(metadata=None, with_nac=False):
        inputs = AttributeDict(
            {
                "structure": generate_structure(),
                "metadata": metadata or {},
            }
        )
        if with_nac:
            inputs.nac_params = generate_nac_params()
        return inputs

    return _generate_inputs


def test_phono3py_with_nac(
    fixture_sandbox,
    fixture_code,
    generate_calc_job,
    generate_inputs,
    generate_settings,
):
    """Test a phonopy calculation."""
    entry_point_calc_job = "phonoxpy.phono3py"

    inputs = generate_inputs(
        metadata={"options": {"resources": {"tot_num_mpiprocs": 1}}}, with_nac=True
    )
    inputs.update(
        {
            "settings": generate_settings(mesh=50, isotope=True),
            "code": fixture_code(entry_point_calc_job),
        }
    )

    ph_settings = {}
    _setup_phono3py_calculation_keyset4(
        ph_settings, inputs["settings"], run_ltc=True, run_fc=True
    )
    assert set(ph_settings) == set(("mesh", "isotope"))

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(
        ("phono3py.yaml", "fc2.hdf5", "fc3.hdf5")
    )
    assert set(calc_info.codes_info[0].cmdline_params) == set(
        ("-c", "phono3py_params.yaml.xz", "--sym-fc", "--compact-fc", "--nac")
    )


def test_phono3py_fc(
    fixture_sandbox,
    fixture_code,
    generate_calc_job,
    generate_inputs,
    generate_settings,
):
    """Test a phonopy calculation."""
    entry_point_calc_job = "phonoxpy.phono3py"

    inputs = generate_inputs(
        metadata={"options": {"resources": {"tot_num_mpiprocs": 1}}}
    )
    inputs.update(
        {
            "settings": generate_settings(),
            "code": fixture_code(entry_point_calc_job),
        }
    )

    ph_settings = {}
    _setup_phono3py_calculation_keyset4(ph_settings, inputs["settings"], run_fc=True)
    assert not ph_settings

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(
        ("phono3py.yaml", "fc2.hdf5", "fc3.hdf5")
    )
    assert set(calc_info.codes_info[0].cmdline_params) == set(
        ("-c", "phono3py_params.yaml.xz", "--sym-fc", "--compact-fc")
    )


def test_phono3py_ltc_lbte(
    fixture_sandbox,
    fixture_code,
    generate_calc_job,
    generate_inputs,
    generate_settings,
    generate_fc3_filedata,
    generate_fc2_filedata,
):
    """Test a phonopy calculation."""
    entry_point_calc_job = "phonoxpy.phono3py"

    inputs = generate_inputs(
        metadata={"options": {"resources": {"tot_num_mpiprocs": 1}}}
    )
    inputs.update(
        {
            "settings": generate_settings(
                mesh=50,
                phonon_supercell_matrix=[2, 2, 2],
                ts=[300, 400, 500],
                lbte=True,
            ),
            "code": fixture_code(entry_point_calc_job),
            "fc2": generate_fc2_filedata(),
            "fc3": generate_fc3_filedata(),
        }
    )

    ph_settings = {}
    _setup_phono3py_calculation_keyset4(ph_settings, inputs["settings"], run_ltc=True)
    assert set(ph_settings) == set(("mesh", "lbte", "ts"))

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(("phono3py.yaml", "kappa-*.hdf5"))
    assert set(calc_info.codes_info[0].cmdline_params) == set(
        (
            "-c",
            "phono3py_params.yaml.xz",
            "--fc2",
            "--fc3",
            "--mesh",
            "50.0",
            "--lbte",
            "--ts",
            "300",
            "400",
            "500",
        )
    )


def test_phono3py_ltc(
    fixture_sandbox,
    fixture_code,
    generate_calc_job,
    generate_inputs,
    generate_settings,
    generate_fc3_filedata,
    generate_fc2_filedata,
):
    """Test a phonopy calculation."""
    entry_point_calc_job = "phonoxpy.phono3py"

    inputs = generate_inputs(
        metadata={"options": {"resources": {"tot_num_mpiprocs": 1}}}
    )
    inputs.update(
        {
            "settings": generate_settings(
                mesh=50, phonon_supercell_matrix=[2, 2, 2], ts=[300, 400]
            ),
            "code": fixture_code(entry_point_calc_job),
            "fc2": generate_fc2_filedata(),
            "fc3": generate_fc3_filedata(),
        }
    )

    ph_settings = {}
    _setup_phono3py_calculation_keyset4(ph_settings, inputs["settings"], run_ltc=True)
    assert set(ph_settings) == set(("mesh", "ts"))

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(("phono3py.yaml", "kappa-*.hdf5"))
    assert set(calc_info.codes_info[0].cmdline_params) == set(
        (
            "-c",
            "phono3py_params.yaml.xz",
            "--fc2",
            "--fc3",
            "--mesh",
            "50.0",
            "--br",
            "--ts",
            "300",
            "400",
        )
    )


def test_phono3py_ltc_with_isotope(
    fixture_sandbox,
    fixture_code,
    generate_calc_job,
    generate_inputs,
    generate_settings,
    generate_fc3_filedata,
    generate_fc2_filedata,
):
    """Test a phonopy calculation."""
    entry_point_calc_job = "phonoxpy.phono3py"

    inputs = generate_inputs(
        metadata={"options": {"resources": {"tot_num_mpiprocs": 1}}}
    )
    inputs.update(
        {
            "settings": generate_settings(
                mesh=[21, 21, 21], isotope=True, phonon_supercell_matrix=[2, 2, 2]
            ),
            "code": fixture_code(entry_point_calc_job),
            "fc2": generate_fc2_filedata(),
            "fc3": generate_fc3_filedata(),
        }
    )

    ph_settings = {}
    _setup_phono3py_calculation_keyset4(ph_settings, inputs["settings"], run_ltc=True)
    assert set(ph_settings) == set(("mesh", "isotope"))

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(("phono3py.yaml", "kappa-*.hdf5"))
    assert set(calc_info.codes_info[0].cmdline_params) == set(
        (
            "-c",
            "phono3py_params.yaml.xz",
            "--fc2",
            "--fc3",
            "--mesh",
            "21",
            "21",
            "21",
            "--br",
            "--ts",
            "300",
            "--isotope",
        )
    )
