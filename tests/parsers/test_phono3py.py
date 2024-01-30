"""Test phono3py parser."""

import pytest
from aiida import orm
from aiida.common import AttributeDict


@pytest.fixture
def generate_inputs(generate_structure, generate_nac_params):
    """Return only those inputs that the parser will expect to be there."""

    def _generate_inputs(metadata=None):
        return AttributeDict(
            {
                "structure": generate_structure(),
                "metadata": metadata or {},
                "nac_params": generate_nac_params(),
            }
        )

    return _generate_inputs


def test_phono3py_default(
    fixture_localhost,
    generate_calc_job_node,
    generate_parser,
    generate_inputs,
    generate_settings,
    num_regression,
):
    """Test a phonopy calculation."""
    import h5py

    name = "default"
    entry_point_calc_job = "phonoxpy.phono3py"
    entry_point_parser = "phonoxpy.phono3py"

    inputs = generate_inputs()
    inputs["settings"] = generate_settings(phonon_supercell_matrix=[2, 2, 2])
    node = generate_calc_job_node(
        entry_point_name=entry_point_calc_job,
        computer=fixture_localhost,
        test_name=name,
        inputs=inputs,
    )
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert not orm.Log.objects.get_logs_for(node), [
        log.message for log in orm.Log.objects.get_logs_for(node)
    ]

    output_keys = ["version", "fc2", "fc3"]
    assert set(list(results)) == set(output_keys)

    for key in [
        "version",
    ]:
        assert key in results

    with results["fc2"].open(mode="rb") as f2, results["fc3"].open(mode="rb") as f3:
        with h5py.File(f2) as f_fc2, h5py.File(f3) as f_fc3:
            num_regression.check(
                {
                    "force_constants": f_fc2["force_constants"][:].ravel(),
                    "fc3": f_fc3["fc3"][:].ravel(),
                }
            )


def test_phono3py_ltc_default(
    fixture_localhost,
    generate_calc_job_node,
    generate_parser,
    generate_inputs,
    generate_settings,
    generate_fc3_filedata,
    generate_fc2_filedata,
    num_regression,
):
    """Test a phonopy calculation."""
    import h5py

    name = "ltc-default"
    entry_point_calc_job = "phonoxpy.phono3py"
    entry_point_parser = "phonoxpy.phono3py"

    inputs = generate_inputs()
    inputs["settings"] = generate_settings(
        supercell_matrix=[2, 2, 2], phonon_supercell_matrix=[4, 4, 4], mesh=30
    )
    inputs["fc3"] = generate_fc3_filedata(structure_id="NaCl-64")
    inputs["fc2"] = generate_fc2_filedata(structure_id="NaCl-512")
    node = generate_calc_job_node(
        entry_point_name=entry_point_calc_job,
        computer=fixture_localhost,
        test_name=name,
        inputs=inputs,
    )
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert not orm.Log.objects.get_logs_for(node), [
        log.message for log in orm.Log.objects.get_logs_for(node)
    ]

    output_keys = ["version", "ltc"]
    assert set(list(results)) == set(output_keys)

    for key in [
        "version",
    ]:
        assert key in results

    with results["ltc"].open(mode="rb") as f:
        with h5py.File(f) as f_ltc:
            num_regression.check(
                {
                    "ltc": f_ltc["kappa"][:].ravel(),
                }
            )
