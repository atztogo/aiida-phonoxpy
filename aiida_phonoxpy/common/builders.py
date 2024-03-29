"""Utilities related to process builder or inputs dist."""

from aiida.orm import (
    Bool,
    Code,
    Str,
    load_group,
    StructureData,
    KpointsData,
    Dict,
    RemoteData,
)
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory

from aiida_phonoxpy.utils.utils import get_kpoints_data


def get_workchain_inputs(calculator_inputs, structure, label=None, ctx=None):
    """Return builder inputs of a calculation."""
    code = _get_code(calculator_inputs)
    plugin_name = code.get_input_plugin_name()
    if plugin_name == "vasp.vasp":
        return _get_vasp_vasp_workchain_inputs(
            calculator_inputs, structure, code, label
        )
    elif plugin_name == "quantumespresso.pw":
        return _get_qe_pw_workchain_inputs(calculator_inputs, structure, code, label)
    elif plugin_name == "quantumespresso.ph":
        return _get_qe_ph_workchain_inputs(calculator_inputs, code, label, ctx)
    else:
        raise RuntimeError("Code could not be found.")


def get_import_workchain_inputs(calculator_inputs, label=None, ctx=None):
    """Return builder inputs of an import calculation."""
    code = _get_code(calculator_inputs)
    plugin_name = code.get_input_plugin_name()
    if plugin_name == "vasp.vasp":
        return _get_vasp_import_workchain_inputs(calculator_inputs, code, label)


def _get_code(calculator_inputs: dict):
    code = None
    if "code" in calculator_inputs:
        code = calculator_inputs["code"]
    elif "code_string" in calculator_inputs:
        code = Code.get_from_string(calculator_inputs["code_string"])
    if code is None:
        for namespace in ("pw", "ph"):
            if namespace in calculator_inputs:
                code = _get_code(calculator_inputs[namespace])
                if code is not None:
                    break
    return code


def _get_vasp_vasp_workchain_inputs(calculator_inputs, structure, code, label):
    inputs = _get_vasp_import_workchain_inputs(calculator_inputs, code, label)
    inputs.update(
        {
            "parameters": _get_parameters_Dict(calculator_inputs),
            "kpoints": _get_kpoints_data(calculator_inputs, structure),
            "structure": structure,
        }
    )
    return inputs


def _get_vasp_import_workchain_inputs(calculator_inputs, code, label):
    inputs = {
        "settings": _get_vasp_settings(calculator_inputs),
        "clean_workdir": Bool(False),
        "code": code,
    }

    if isinstance(calculator_inputs["options"], dict):
        inputs["options"] = Dict(dict=calculator_inputs["options"])
    elif isinstance(calculator_inputs["options"], Dict):
        inputs["options"] = calculator_inputs["options"]
    else:
        raise TypeError("options has to have dict or Dict type.")

    if isinstance(calculator_inputs["potential_family"], str):
        inputs["potential_family"] = Str(calculator_inputs["potential_family"])
    elif isinstance(calculator_inputs["potential_family"], Str):
        inputs["potential_family"] = calculator_inputs["potential_family"]
    else:
        raise TypeError("potential_family has to have str or Str type.")

    if isinstance(calculator_inputs["potential_mapping"], dict):
        inputs["potential_mapping"] = Dict(dict=calculator_inputs["potential_mapping"])
    elif isinstance(calculator_inputs["potential_mapping"], Dict):
        inputs["potential_mapping"] = calculator_inputs["potential_mapping"]
    else:
        raise TypeError("potential_mapping has to have dict or Dict type.")

    if "restart_folder" in calculator_inputs:
        inputs["restart_folder"] = calculator_inputs["restart_folder"]

    if label:
        inputs["metadata"] = {"label": label}
    return inputs


def _get_qe_pw_inputs(
    calculator_inputs_pw: dict, structure: StructureData, code: Code, label: str
) -> dict:
    pseudos = _get_qe_pseudos(calculator_inputs_pw, structure)
    if "metadata" in calculator_inputs_pw:
        metadata = calculator_inputs_pw["metadata"]
    else:
        metadata = {}
    if label:
        metadata["label"] = label
    pw = {
        "metadata": metadata,
        "parameters": _get_parameters_Dict(calculator_inputs_pw),
        "structure": structure,
        "pseudos": pseudos,
        "code": code,
    }

    return pw


def _get_qe_ph_inputs(
    calculator_inputs_ph: dict, code: Code, label: str, remote_folder: RemoteData
) -> dict:
    qpoints = KpointsData()
    qpoints.set_kpoints_mesh([1, 1, 1], offset=[0, 0, 0])
    if "metadata" in calculator_inputs_ph:
        metadata = calculator_inputs_ph["metadata"]
    else:
        metadata = {}
    if label:
        metadata["label"] = label
    ph = {
        "metadata": metadata,
        "qpoints": qpoints,
        "parameters": _get_parameters_Dict(calculator_inputs_ph),
        "parent_folder": remote_folder,
        "code": code,
    }
    return ph


def _get_qe_pw_workchain_inputs(
    calculator_inputs: dict, structure: StructureData, code: Code, label: str
) -> dict:
    kpoints = _get_kpoints_data(calculator_inputs, structure)
    pw = _get_qe_pw_inputs(calculator_inputs["pw"], structure, code, label)
    workchain_inputs = {
        "kpoints": kpoints,
        "pw": pw,
    }
    return workchain_inputs


def _get_qe_ph_workchain_inputs(
    calculator_inputs: dict, code: Code, label: str, ctx: AttributeDict
) -> dict:
    ph = _get_qe_ph_inputs(
        calculator_inputs["ph"],
        code,
        label,
        ctx.nac_params_calcs[0].outputs.remote_folder,
    )
    workchain_inputs = {"ph": ph}
    return workchain_inputs


def get_calculator_process(plugin_name):
    """Return WorkChain or CalcJob."""
    if plugin_name == "vasp.vasp":
        return WorkflowFactory(plugin_name)
    elif plugin_name in ("quantumespresso.pw", "quantumespresso.ph"):
        return WorkflowFactory(plugin_name + ".base")
    else:
        raise RuntimeError("Code could not be found.")


def _get_kpoints_data(calculator_inputs, structure):
    """Return KpointsData."""
    if "kpoints" in calculator_inputs.keys():
        assert isinstance(calculator_inputs["kpoints"], KpointsData)
        return calculator_inputs["kpoints"]
    return get_kpoints_data(calculator_inputs, structure)


def _get_qe_pseudos(calculator_inputs: dict, structure: StructureData):
    if "pseudos" in calculator_inputs:
        return calculator_inputs["pseudos"]
    else:
        family = load_group(calculator_inputs["pseudo_family_string"])
        pseudos = family.get_pseudos(structure=structure)
        return pseudos


def _get_parameters_Dict(calculator_inputs):
    """Return parameters for inputs.parameters.

    If calculator_inputs["parameters"] is already a Dict,
    a new Dict will not be made, and just it will be returned.

    """
    if isinstance(calculator_inputs["parameters"], dict):
        return Dict(dict=calculator_inputs["parameters"])
    elif isinstance(calculator_inputs["parameters"], Dict):
        return calculator_inputs["parameters"]
    else:
        raise TypeError("parameters has to have dict or Dict type.")


def _get_vasp_settings(calculator_inputs):
    """Update VASP settings.

    If no update of settings and calculator_inputs["settings"] is already a Dict,
    a new Dict will not be made, and just it will be returned.

    """
    updated = False
    if "settings" in calculator_inputs.keys():
        if isinstance(calculator_inputs["settings"], dict):
            settings = calculator_inputs["settings"]
        elif isinstance(calculator_inputs["settings"], Dict):
            settings = calculator_inputs["settings"].get_dict()
        else:
            raise TypeError("settings has to have dict or Dict type.")
    else:
        settings = {}
    if "parser_settings" in calculator_inputs.keys():
        settings["parser_settings"] = calculator_inputs["parser_settings"]
        updated = True
    if (
        "parser_settings" not in settings
        or "add_forces" not in settings["parser_settings"]
    ):
        settings["parser_settings"].update({"add_forces": True})
        updated = True

    assert settings

    if updated or isinstance(calculator_inputs["settings"], dict):
        return Dict(dict=settings)
    else:
        return calculator_inputs["settings"]


def get_plugin_names(calculator_inputs: dict) -> list:
    """Return plugin names of calculators."""
    codes = []
    if "steps" in calculator_inputs:
        for step in calculator_inputs["steps"]:
            codes.append(_get_code(step))
    else:
        codes.append(_get_code(calculator_inputs))

    plugin_names = []
    for code in codes:
        plugin_names.append(code.get_input_plugin_name())

    return plugin_names
