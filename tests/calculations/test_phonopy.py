"""Test phono3py parser."""
import pytest
from aiida.common import AttributeDict


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


def test_phonopy(
    fixture_sandbox,
    fixture_code,
    generate_calc_job,
    generate_inputs,
    generate_settings,
):
    """Test a phonopy calculation."""
    entry_point_calc_job = "phonoxpy.phonopy"

    inputs = generate_inputs(
        metadata={"options": {"resources": {"tot_num_mpiprocs": 1}}}
    )
    inputs.update(
        {
            "settings": generate_settings(),
            "code": fixture_code(entry_point_calc_job),
        }
    )

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(("phonopy.yaml", "force_constants.hdf5"))
    assert not calc_info.local_copy_list
    assert set(calc_info.codes_info[0].cmdline_params) == set(
        (
            "-c",
            "--sym-fc",
            "phonopy_params.yaml.xz",
            "--writefc",
            "--writefc-format=hdf5",
        )
    )


def test_phonopy_fc_calculator(
    fixture_sandbox,
    fixture_code,
    generate_calc_job,
    generate_inputs,
    generate_settings,
):
    """Test a phonopy calculation."""
    entry_point_calc_job = "phonoxpy.phonopy"

    inputs = generate_inputs(
        metadata={"options": {"resources": {"tot_num_mpiprocs": 1}}}
    )
    inputs.update(
        {
            "settings": generate_settings(fc_calculator="alm"),
            "code": fixture_code(entry_point_calc_job),
        }
    )

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(("phonopy.yaml", "force_constants.hdf5"))
    assert not calc_info.local_copy_list
    assert set(calc_info.codes_info[0].cmdline_params) == set(
        (
            "-v",
            "-c",
            "phonopy_params.yaml.xz",
            "--writefc",
            "--writefc-format=hdf5",
            "--alm",
        )
    )


def test_phonopy_fc_input(
    fixture_sandbox,
    fixture_code,
    generate_calc_job,
    generate_inputs,
    generate_settings,
    generate_fc_filedata,
):
    """Test a phonopy calculation."""
    entry_point_calc_job = "phonoxpy.phonopy"

    inputs = generate_inputs(
        metadata={"options": {"resources": {"tot_num_mpiprocs": 1}}}
    )
    inputs.update(
        {
            "settings": generate_settings(),
            "code": fixture_code(entry_point_calc_job),
            "force_constants": generate_fc_filedata(),
        }
    )

    calc_info = generate_calc_job(fixture_sandbox, entry_point_calc_job, inputs)
    assert set(calc_info.retrieve_list) == set(("phonopy.yaml",))
    assert calc_info.local_copy_list[0][2] == "force_constants.hdf5"
    assert set(calc_info.codes_info[0].cmdline_params) == set(
        ("-c", "phonopy_params.yaml.xz", "--readfc", "--readfc-format=hdf5")
    )
