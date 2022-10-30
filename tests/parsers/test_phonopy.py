"""Test phonopy parser."""
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


def test_phonopy_default(
    fixture_localhost,
    generate_calc_job_node,
    generate_parser,
    generate_inputs,
    data_regression,
    num_regression,
):
    """Test a phonopy calculation."""
    import h5py

    name = "default"
    entry_point_calc_job = "phonoxpy.phonopy"
    entry_point_parser = "phonoxpy.phonopy"

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

    for key in [
        "force_constants",
        "projected_dos",
        "thermal_properties",
        "band_structure",
        "version",
    ]:
        assert key in results

    data_regression.check(
        {
            "pdos": results["projected_dos"].attributes,
            "thermal_properties": results["thermal_properties"].attributes,
            "band_structure": results["band_structure"].attributes,
        }
    )

    with results["force_constants"].open(mode="rb") as fc:
        with h5py.File(fc) as f_fc:
            num_regression.check(
                {
                    "force_constants": f_fc["force_constants"][:].ravel(),
                }
            )

    # Old test when force_constants was ArrayData.
    # num_regression.check(
    #     {
    #         "force_constants": results["force_constants"]
    #         .get_array("force_constants")
    #         .ravel()
    #     }
    # )
