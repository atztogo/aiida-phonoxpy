"""Test phono3py parser."""
import pytest
from aiida.common import AttributeDict
from aiida_phonoxpy.utils.utils import (
    _setup_phono3py_calculation_keyset4,
    _setup_phono3py_calculation_keyset5,
)


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
    _setup_phono3py_calculation_keyset4(ph_settings, inputs["settings"])
    assert not ph_settings

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(
        ("phono3py.yaml", "fc2.hdf5", "fc3.hdf5")
    )
    assert set(calc_info.codes_info[0].cmdline_params) == set(
        ("-c", "phono3py_params.yaml.xz", "--sym-fc", "--compact-fc")
    )


def test_phono3py_fc_fc_calculator(
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
            "settings": generate_settings(fc_calculator="alm"),
            "code": fixture_code(entry_point_calc_job),
        }
    )

    ph_settings = {}
    _setup_phono3py_calculation_keyset4(ph_settings, inputs["settings"])
    assert "fc_calculator" in ph_settings

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(
        ("phono3py.yaml", "fc2.hdf5", "fc3.hdf5")
    )
    assert set(calc_info.codes_info[0].cmdline_params) == set(
        ("-v", "--alm", "-c", "phono3py_params.yaml.xz", "--compact-fc")
    )


def test_phono3py_fc_fc_calculator_options(
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
            "settings": generate_settings(
                fc_calculator="alm", fc_calculator_options="cutoff = 5"
            ),
            "code": fixture_code(entry_point_calc_job),
        }
    )

    ph_settings = {}
    _setup_phono3py_calculation_keyset4(ph_settings, inputs["settings"])
    assert "fc_calculator" in ph_settings
    assert "fc_calculator_options" in ph_settings

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(
        ("phono3py.yaml", "fc2.hdf5", "fc3.hdf5")
    )
    assert set(calc_info.codes_info[0].cmdline_params) == set(
        (
            "-v",
            "--alm",
            "--fc-calculator-options",
            "cutoff = 5",
            "-c",
            "phono3py_params.yaml.xz",
            "--compact-fc",
        )
    )


@pytest.mark.parametrize(
    "opt_key,opt_val",
    [
        ("cutoff_fc3", 20),
        ("mass_variances", [1e-5, 1e-6]),
        ("pinv_cutoff", 1e-8),
        ("pinv_solver", 1),
        ("reducible_colmat", True),
        ("sigma", 0.1),
        ("sigma", [0.1, 0.2]),
    ],
)
def test_phono3py_ltc_simple_options(
    fixture_sandbox,
    fixture_code,
    generate_calc_job,
    generate_inputs,
    generate_settings,
    generate_fc3_filedata,
    generate_fc2_filedata,
    opt_key,
    opt_val,
):
    """Test a phonopy calculation."""
    entry_point_calc_job = "phonoxpy.phono3py"

    inputs = generate_inputs(
        metadata={"options": {"resources": {"tot_num_mpiprocs": 1}}}
    )
    settings = {opt_key: opt_val}
    inputs.update(
        {
            "settings": generate_settings(**settings),
            "code": fixture_code(entry_point_calc_job),
            "fc2": generate_fc2_filedata(),
            "fc3": generate_fc3_filedata(),
        }
    )

    ph_settings = {}
    _setup_phono3py_calculation_keyset5(ph_settings, inputs["settings"])
    assert opt_key in ph_settings
    if opt_key == "sigma":
        if isinstance(opt_val, list):
            option_set = ("--sigma",) + tuple([f"{val}" for val in opt_val])
        else:
            option_set = ("--sigma", "0.1")
    elif opt_key == "reducible_colmat":
        option_set = (f"--{opt_key.replace('_', '-')}",)
    elif opt_key == "mass_variances":
        option_set = (f"--{opt_key.replace('_', '-')}",) + tuple(
            [f"{val}" for val in opt_val]
        )
    else:
        option_set = (f"--{opt_key.replace('_', '-')}", f"{opt_val}")

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(("phono3py.yaml", "kappa-*.hdf5"))
    assert set(calc_info.codes_info[0].cmdline_params) == set(
        (
            "-c",
            "phono3py_params.yaml.xz",
            "--fc2",
            "--fc3",
            "--mesh",
            "30",
            "--br",
            "--ts",
            "300",
        )
        + option_set
    )


def test_phono3py_with_ltc_nac(
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
        metadata={"options": {"resources": {"tot_num_mpiprocs": 1}}}, with_nac=True
    )
    inputs.update(
        {
            "settings": generate_settings(
                mesh=100,
                phonon_supercell_matrix=[2, 2, 2],
            ),
            "code": fixture_code(entry_point_calc_job),
            "fc2": generate_fc2_filedata(),
            "fc3": generate_fc3_filedata(),
        }
    )

    ph_settings = {}
    _setup_phono3py_calculation_keyset5(ph_settings, inputs["settings"])
    assert set(ph_settings) == set(("mesh",))

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(("phono3py.yaml", "kappa-*.hdf5"))
    assert set(calc_info.codes_info[0].cmdline_params) == set(
        (
            "-c",
            "phono3py_params.yaml.xz",
            "--nac",
            "--fc2",
            "--fc3",
            "--mesh",
            "100.0",
            "--br",
            "--ts",
            "300",
        )
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
    _setup_phono3py_calculation_keyset5(ph_settings, inputs["settings"])
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
    _setup_phono3py_calculation_keyset5(ph_settings, inputs["settings"])
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
    _setup_phono3py_calculation_keyset5(ph_settings, inputs["settings"])
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


def test_phono3py_ltc_with_grg(
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
                mesh=100, grg=True, phonon_supercell_matrix=[2, 2, 2]
            ),
            "code": fixture_code(entry_point_calc_job),
            "fc2": generate_fc2_filedata(),
            "fc3": generate_fc3_filedata(),
        }
    )

    ph_settings = {}
    _setup_phono3py_calculation_keyset5(ph_settings, inputs["settings"])
    assert set(ph_settings) == set(("mesh", "grg"))

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(("phono3py.yaml", "kappa-*.hdf5"))

    ref = (
        "-c",
        "phono3py_params.yaml.xz",
        "--fc2",
        "--fc3",
        "--mesh",
        100.0,
        "--br",
        "--ts",
        "300",
        "--grg",
    )

    _compare_sets(ref, calc_info.codes_info[0].cmdline_params)


def _compare_sets(ref, data):
    assert len(ref) == len(data)
    num_vals = []
    for val in ref:
        if isinstance(val, str):
            assert val in data
        else:
            num_vals.append(val)

    for val in num_vals:
        is_found = False
        for dval in data:
            try:
                if float(dval) == pytest.approx(val):
                    is_found = True
                    break
            except ValueError:
                pass
        assert is_found, f"{val} didn't match any data in {data}."
