"""Test phono3py parser."""
import pytest
from aiida import orm
from aiida.common import AttributeDict


@pytest.fixture
def generate_inputs(generate_structure, generate_nac_params, generate_settings):
    """Return only those inputs that the parser will expect to be there."""

    def _generate_inputs(metadata=None):
        return AttributeDict(
            {
                "structure": generate_structure(),
                "settings": generate_settings(),
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
    num_regression,
):
    """Test a phonopy calculation."""
    import h5py

    name = "default"
    entry_point_calc_job = "phonoxpy.phono3py"
    entry_point_parser = "phonoxpy.phono3py"

    node = generate_calc_job_node(
        entry_point_name=entry_point_calc_job,
        computer=fixture_localhost,
        test_name=name,
        inputs=generate_inputs(),
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

    with results["fc2"].open() as f:
        path_fc2 = f.name
    with results["fc3"].open() as f:
        path_fc3 = f.name

    with h5py.File(path_fc2, "r") as f_fc2, h5py.File(path_fc3, "r") as f_fc3:
        num_regression.check(
            {
                "force_constants": f_fc2["force_constants"][:].ravel(),
                "fc3": f_fc3["fc3"][:].ravel(),
            }
        )