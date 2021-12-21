"""Utilities related to process builder or inputs dist."""
import copy

from aiida.orm import Bool, Code, Str, load_group
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_phonoxpy.common.utils import get_kpoints_data

KpointsData = DataFactory("array.kpoints")
Dict = DataFactory("dict")


def get_workchain_inputs(calculator_inputs, structure, label=None, ctx=None):
    """Return builder inputs of a calculation."""
    if "code" in calculator_inputs.keys():
        code = calculator_inputs["code"]
    else:
        code = Code.get_from_string(calculator_inputs["code_string"])
    plugin_name = code.get_input_plugin_name()
    if plugin_name == "vasp.vasp":
        return _get_vasp_vasp_workchain_inputs(
            calculator_inputs, structure, code, label
        )
    elif plugin_name == "quantumespresso.pw":
        return _get_quantumespresso_pw_workchain_inputs(
            calculator_inputs, structure, code, label
        )
    elif plugin_name == "quantumespresso.ph":
        return _get_quantumespresso_ph_workchain_inputs(
            calculator_inputs, code, label, ctx
        )
    else:
        raise RuntimeError("Code could not be found.")


def _get_vasp_vasp_workchain_inputs(calculator_inputs, structure, code, label):
    kpoints = _get_kpoints_data(calculator_inputs, structure)
    if isinstance(calculator_inputs["options"], dict):
        options = Dict(dict=calculator_inputs["options"])
    elif isinstance(calculator_inputs["options"], Dict):
        options = calculator_inputs["options"]
    else:
        raise TypeError("options has to have dict or Dict type.")

    if isinstance(calculator_inputs["potential_family"], str):
        potential_family = Str(calculator_inputs["potential_family"])
    elif isinstance(calculator_inputs["potential_family"], Str):
        potential_family = calculator_inputs["potential_family"]
    else:
        raise TypeError("potential_family has to have str or Str type.")

    if isinstance(calculator_inputs["potential_mapping"], dict):
        potential_mapping = Dict(dict=calculator_inputs["potential_mapping"])
    elif isinstance(calculator_inputs["potential_mapping"], Dict):
        potential_mapping = calculator_inputs["potential_mapping"]
    else:
        raise TypeError("potential_mapping has to have dict or Dict type.")

    workchain_inputs = {
        "options": options,
        "parameters": _get_parameters_Dict(calculator_inputs),
        "settings": _get_vasp_settings(calculator_inputs),
        "kpoints": kpoints,
        "clean_workdir": Bool(False),
        "structure": structure,
        "code": code,
        "potential_family": potential_family,
        "potential_mapping": potential_mapping,
    }
    if label:
        workchain_inputs["metadata"] = {"label": label}
    return workchain_inputs


def _get_quantumespresso_pw_workchain_inputs(calculator_inputs, structure, code, label):
    kpoints = _get_kpoints_data(calculator_inputs, structure)
    family = load_group(calculator_inputs["pseudo_family_string"])
    pseudos = family.get_pseudos(structure=structure)
    metadata = {"options": calculator_inputs["options"]}
    if label:
        metadata["label"] = label
    pw = {
        "metadata": metadata,
        "parameters": _get_parameters_Dict(calculator_inputs),
        "structure": structure,
        "pseudos": pseudos,
        "code": code,
    }
    workchain_inputs = {
        "kpoints": kpoints,
        "pw": pw,
    }
    return workchain_inputs


def _get_quantumespresso_ph_workchain_inputs(calculator_inputs, code, label, ctx):
    qpoints = KpointsData()
    qpoints.set_kpoints_mesh([1, 1, 1], offset=[0, 0, 0])
    metadata = {"options": calculator_inputs["options"]}
    if label:
        metadata["label"] = label
    ph = {
        "metadata": metadata,
        "qpoints": qpoints,
        "parameters": _get_parameters_Dict(calculator_inputs),
        "parent_folder": ctx.nac_params_calcs[0].outputs.remote_folder,
        "code": code,
    }
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


def get_plugin_names(calculator_settings):
    """Return plugin names of calculators."""
    codes = []
    if "steps" in calculator_settings.keys():
        for step in calculator_settings["steps"]:
            if "code_string" in step.keys():
                code = Code.get_from_string(step["code_string"])
            else:
                code = step["code"]
            codes.append(code)
    else:
        if "code_string" in calculator_settings.keys():
            code = Code.get_from_string(calculator_settings["code_string"])
        else:
            code = calculator_settings["code"]
        codes.append(code)

    plugin_names = []
    for code in codes:
        plugin_names.append(code.get_input_plugin_name())

    return plugin_names


def get_vasp_immigrant_inputs(folder_path, calculator_settings, label=None):
    """Return VASP immigrant inputs.

    folder_path : str
        VASP directory path.
    calculator_settings : dict
        aiida-phonopy calculator settings for forces or nac params.

    """
    code = Code.get_from_string(calculator_settings["code_string"])

    if code.get_input_plugin_name() == "vasp.vasp":
        inputs = {}
        inputs["code"] = code
        inputs["folder_path"] = Str(folder_path)
        if "settings" in calculator_settings:
            settings = copy.deepcopy(calculator_settings["settings"])
        else:
            settings = {}
        if "parser_settings" in calculator_settings:
            if "parser_settings" in settings:
                settings["parser_settings"].update(
                    calculator_settings["parser_settings"]
                )
            else:
                settings["parser_settings"] = calculator_settings["parser_settings"]
        if settings:
            inputs["settings"] = Dict(dict=settings)
        if "options" in calculator_settings:
            inputs["options"] = Dict(dict=calculator_settings["options"])
        if "metadata" in calculator_settings:
            inputs["metadata"] = calculator_settings["metadata"]
            if label:
                inputs["metadata"]["label"] = label
        elif label:
            inputs["metadata"] = {"label": label}
        if "potential_family" in calculator_settings:
            inputs["potential_family"] = Str(calculator_settings["potential_family"])
        if "potential_mapping" in calculator_settings:
            inputs["potential_mapping"] = Dict(
                dict=calculator_settings["potential_mapping"]
            )
    else:
        raise RuntimeError("Code could not be found.")

    return inputs
