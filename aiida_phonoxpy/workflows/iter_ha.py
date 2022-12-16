"""Run phonon calculations iteratively at temperature."""
import os
import tempfile
import h5py
import shutil
import numpy as np
from aiida.engine import WorkChain, calcfunction, if_, while_
from aiida.orm import (
    ArrayData,
    Bool,
    Code,
    Dict,
    Float,
    SinglefileData,
    Group,
    Int,
    QueryBuilder,
    load_node,
)
from phonopy import Phonopy

from aiida_phonoxpy.utils.utils import (
    get_displacements_from_phonopy_wc,
    phonopy_atoms_from_structure,
)
from aiida_phonoxpy.workflows.phonopy import PhonopyWorkChain

from phonopy.units import EvTokJmol, THzToEv
from phonopy.file_IO import write_force_constants_to_hdf5
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import (
    Primitive,
    convert_to_phonopy_primitive,
)
from phonopy.harmonic.force_constants import distribute_force_constants_by_translations

"""

Parameters in get_random_displacements and collect_dataset
----------------------------------------------------------
To control how to include previous snapshots, several parameters
can be used.

1) number_of_steps_for_fitting : Int
    Maximum number of previous phonon calculation steps included. When
    number of the previous phonon calculations is smaller than this
    number, all the previous phonon calculations are included. But
    this number is used for (2) in any case.
2) linear_decay : True
    This will be an option, but currently always True.
    This controls weights of previous phonon calculations. One previous
    phonon calculation is included 100%. (number_of_steps_for_fitting + 1)
    previous phonon calculation is not include. Between them, those are
    included with linear scalings. The snapshots in each phonon calculation
    are included from the first element of the list to the specified
    number, i.e., [:num_included_snapshots].
3) include_ratio : Float
    After collecting snapshots in (2), all the snapshots are sorted by
    total energies. Then only lowest energy snapshots with 'include_ratio'
    are included to calculate force constants by fitting using ALM.
4) random_seed : Int
    Using force constants created in (3), phonons are calculated at
    commensurate points, and these phonons are used to generate atomic
    displacements by sampling harmonic oscillator distribution function.
    For this 'random_seed' is used when provided.

"""


class IterHarmonicApprox(WorkChain):
    """Workchain for harmonic force constants by iterative approach.

    By default, the calculation starts with normal phonon calculation, i.e., in
    this context, which corresponds to roughly 0K force constants. Then the
    iteration loop starts. The first run is the iteration-1. The iteration stops
    after finishing that of max_iteration. Each phonon calculation is named
    'step'.

    Steps
    -----
    0. Initial phonon calculation normally at 0K. This phonon calculation is
       controled by inputs.settings similarly to PhonopyWorkchain. Therefore
       inputs.force_constants, temperature in inputs.settings['temperature']
       (not inputs.temperature), and inputs.settings['number_of_snapshots'] (not
       inputs.number_of_snapshots) are simultaneously set, random displacements
       at the temperature are generated.
    1. First phonon calculation at specified temperature. Random displacements
       are created from step-0.
    2. Second phonon calculation at specified temperature. Random displacements
       are created from step-1. When include_initial_phonon=True, step-0 is also
       included in the dataset to generate random displacements.
    3. Third phonon calculation at specified temperature. Random displacements
       are created from steps 1 and 2 if number_of_snapshots >= 2. Otherwise
       only the result from step-2 is used. When include_initial_phonon=True,
       step-0 is included.
    4. Fourth phonon calculation at specified temperature. Random displacements
       are created from number_of_snapshots previous existing steps excluding
       step-0 unless include_initial_phonon=True.
    *. Continue until iteration number = max_iteration number.

    Manual termination of iteration loop
    ------------------------------------
    It is possible to terminate at the initial point of each iteration. This
    option is not very recommended to use because not reproducible mechanically,
    but can be useful for experimental calculation.

    This is achieved by just creating AiiDA Group whose label is its uuid string
    that ejected by AiiDA, i.e., self.uuid.

    inputs
    ------
    Most of inputs are imported from PhonopyWorkChain. Specific inputs of this
    workchain are as follows:

    max_iteration : Int
        Maximum number of iterations.
    number_of_snapshots : Int
        Number of generated supercell snapshots with random displacements at a
        temperature.
    number_of_steps_for_fitting : Int
        Displacements and respective forces of supercells in the previous
        number_of_steps_for_fitting are used to simultaneously fit to force
        constants.
    temperature : Float
        Temperature (K).
    include_ratio : Float
        How much supercell forces are included from lowest supercell energies.
        Default is 1.0.
    lienar_decay : Bool
        This controls weights of previous phonon calculations. One previous
        phonon calculation is included 100%. (number_of_steps_for_fitting + 1)
        previous phonon calculation is not include. Between them, those are
        included with linear scalings. The snapshots in each phonon calculation
        are included from the first element of the list to the specified number,
        i.e., [:num_included_snapshots]. Default is False.
    random_seed : Int, optional
        Random seed used to sample in canonical ensemble harmonic oscillator
        space. The value must be 32bit unsigned int. Unless specified, random
        seed will not be fixed.
    initial_nodes : Dict, optional
        This gives the initial nodes that contain sets of forces, which are
        provided by PKs or UUIDs.
    include_initial_phonon : Bool, optional
        With False, the initial phonon calculation is only used to generate
        random displacement at specified temperature, and next supercell
        calculations are included in the set of displacement-force datasets as
        emsemble to generate self-consistent force constants at interation
        steps. With True, the supercell calculations of the initial phonon
        calculation are also included in the ensemble. Default is False.
    use_reweighting : Bool, optional
        When True, reweighting of displacement-force dataset by an importance
        sampling is performed in the calculation of force consntants. Default is
        False.

    """

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)
        spec.expose_inputs(
            PhonopyWorkChain,
            exclude=["remote_workdirs", "run_phonopy"],
        )
        spec.input("max_iteration", valid_type=Int, default=lambda: Int(10))
        spec.input("number_of_snapshots", valid_type=Int, default=lambda: Int(100))
        spec.input(
            "number_of_steps_for_fitting", valid_type=Int, default=lambda: Int(4)
        )
        spec.input("temperature", valid_type=Float, default=lambda: Float(300.0))
        spec.input("include_ratio", valid_type=Float, default=lambda: Float(1))
        spec.input("linear_decay", valid_type=Bool, default=lambda: Bool(False))
        spec.input(
            "include_initial_phonon", valid_type=Bool, default=lambda: Bool(False)
        )
        spec.input("random_seed", valid_type=Int, required=False)
        spec.input("initial_nodes", valid_type=Dict, required=False)
        spec.input("use_reweighting", valid_type=Bool, default=lambda: Bool(False))
        spec.outline(
            cls.initialize,
            if_(cls.import_initial_nodes)(cls.set_initial_nodes).else_(
                cls.run_initial_phonon,
            ),
            cls.set_crystal_structures,
            cls.create_dataset_for_fc,
            if_(cls.should_run_remote_phonopy)(
                cls.run_force_constants_calculation_remote,
                cls.set_force_constants,
            ).else_(
                cls.run_force_constants_calculation_local,
            ),
            if_(cls.should_include_initial_phonon)(
                cls.put_initial_phonon_in_history,  # prev_nodes.append(initial_node)
                if_(cls.use_reweighting)(
                    cls.calculate_probability_distribution,  # prev_probs.append(probs)
                ),
            ),
            while_(cls.is_loop_finished)(
                cls.generate_displacements,
                if_(cls.use_reweighting)(
                    cls.calculate_probability_distribution,  # prev_probs.append(probs)
                ),
                cls.increment_iteration_number,
                cls.run_forces_nac_calculations,  # prev_nodes.append(node)
                cls.create_dataset_for_fc,
                if_(cls.should_run_remote_phonopy)(
                    cls.run_force_constants_calculation_remote,
                    cls.set_force_constants,
                ).else_(
                    cls.run_force_constants_calculation_local,
                ),
            ),
            cls.finalize,
        )

    def should_run_remote_phonopy(self):
        """Return boolean for outline."""
        return self.inputs.remote_phonopy

    def import_initial_nodes(self):
        """Return boolean."""
        return "initial_nodes" in self.inputs

    def should_include_initial_phonon(self):
        """Return boolean for outline."""
        return self.inputs.include_initial_phonon

    def use_reweighting(self):
        """Return boolean for outline."""
        return self.inputs.use_reweighting

    def initialize(self):
        """Initialize."""
        self.report("initialize (%s)" % self.uuid)
        self.ctx.iteration = 0
        self.ctx.initial_node = None
        self.ctx.prev_nodes = []
        self.ctx.prev_probs = []
        self.ctx.displacements = None
        self.ctx.force_sets = None
        self.ctx.random_displacements = None
        self.ctx.force_constants = None
        self.ctx.supercell = None
        self.ctx.primitive = None

    def set_initial_nodes(self):
        """Set up initial phonon calculation."""
        self.report("set_initial_phonon")
        node_ids = self.inputs.initial_nodes["nodes"]
        self.ctx.prev_nodes = [load_node(node_id) for node_id in node_ids]

    def run_initial_phonon(self):
        """Launch initial phonon calculation."""
        self.report("run_initial_phonon")
        inputs = self._get_phonopy_inputs(is_initial_phonon=True)
        inputs["subtract_residual_forces"] = Bool(True)
        inputs["metadata"].label = "Initial phonon calculation"
        inputs["metadata"].description = "Initial phonon calculation"
        future = self.submit(PhonopyWorkChain, **inputs)
        self.ctx.initial_node = future
        self.report("{} pk = {}".format(inputs["metadata"].label, future.pk))
        self.to_context(**{"phonon_0": future})

    def set_crystal_structures(self):
        """Set self.ctx.supercell."""
        if "phonon_0" in self.ctx:
            self.report("Found cells in initial phonon calculation.")
            self.ctx.supercell = self.ctx.phonon_0.outputs.supercell
            self.ctx.primitive = self.ctx.phonon_0.outputs.primitive
        elif self.ctx.prev_nodes:
            self.report("Found cells in imported phonon calculation.")
            self.ctx.supercell = self.ctx.prev_nodes[0].outputs.supercell
            self.ctx.primitive = self.ctx.prev_nodes[0].outputs.primitive
        else:
            raise RuntimeError("IterHA is broken.")

    def put_initial_phonon_in_history(self):
        """Set self.ctx.random_displacements."""
        self.report("Put initial phonon calculation in history.")
        self.ctx.prev_nodes.append(self.ctx.initial_node)
        self.ctx.random_displacements = self.ctx.displacements

    def is_loop_finished(self):
        """Check if iteration is over or not.

        ```
        iteration = 0
        force_constants = None
        calculate_initial_forces() -> disps, forces
        while (iteration < max_iteration):
            iteration += 1
            create_dataset(disps, forces, force_constants)
            force_constants_calculation -> force_constants
            generate_random_displacements(force_constants) -> disps
            calculated_forces_with_random_displacements(disps) -> forces
        ```

        iteration=0 : Initial phonon calculation.
        iteration>0 : Phonon calculations with random displacements at temperature

        """
        qb = QueryBuilder()
        qb.append(Group, filters={"label": {"==": self.uuid}})
        if qb.count() == 1:
            self.report(
                "Iteration loop is manually terminated at step %d." % self.ctx.iteration
            )
            return False

        return self.ctx.iteration < self.inputs.max_iteration.value

    def increment_iteration_number(self):
        """Increment iteration."""
        self.ctx.iteration += 1
        self.report(f"IterHA iteration {self.ctx.iteration}")

    def generate_displacements(self):
        """Generate random displacements using force constants.

        The random displacements are generated from phonons and harmonic
        oscillator distribution function of canonical ensemble. The input
        phonons are calculated from force constants calculated from
        forces and displacemens of the supercell snapshots in previous
        phonon calculation steps.

        Set `self.ctx.random_displacements`.

        This method has to come after

        - `run_force_constants_calculation_*`

        """
        phonon_setting_info = self.inputs.settings
        smat = phonon_setting_info["supercell_matrix"]
        ph = Phonopy(
            phonopy_atoms_from_structure(self.inputs.structure),
            supercell_matrix=smat,
            primitive_matrix="auto",
        )
        ph.force_constants = self.ctx.force_constants.get_array("force_constants")

        if "random_seed" in self.inputs:
            random_seed = self.inputs.random_seed.value
        else:
            random_seed = None

        ph.init_random_displacements()
        ph.random_displacements.treat_imaginary_modes()
        disps = ph.get_random_displacements_at_temperature(
            temperature=self.inputs.temperature.value,
            number_of_snapshots=self.inputs.number_of_snapshots.value,
            random_seed=random_seed,
        )
        self.ctx.random_displacements = ArrayData()
        self.ctx.random_displacements.set_array("displacements", disps)

    def calculate_probability_distribution(self):
        """Calculate probability distribution.

        Append to `self.ctx.prev_probs`.

        This method has to come after

        - `run_force_constants_calculation_*`
        - `generate_displacements`

        """
        self.report("Calculate probability distributions.")
        result = get_probability_distribution_data(
            self.ctx.supercell,
            self.ctx.primitive,
            self.ctx.force_constants,
            self.ctx.random_displacements,
            self.inputs.temperature,
        )
        self.ctx.prev_probs.append(result["probability_distributions"])

    def run_forces_nac_calculations(self):
        """Launch phonon calculation with random displacements.

        This method relies on the last iteration:
        - Depends on `self.ctx.random_displacements` made in the last iteration.

        Append `PhonopyWorkChain` node to `self.ctx.prev_nodes`.

        """
        self.report(
            f"run forces and nac calculations at iteration {self.ctx.iteration}"
        )
        inputs = self._get_phonopy_inputs(displacements=self.ctx.random_displacements)
        label = "Phonon calculation %d" % self.ctx.iteration
        inputs["metadata"].label = label
        inputs["metadata"].description = label
        future = self.submit(PhonopyWorkChain, **inputs)
        self.ctx.prev_nodes.append(future)
        self.report("{} pk = {}".format(label, future.pk))
        label = "phonon_%d" % self.ctx.iteration
        self.to_context(**{label: future})

    def create_dataset_for_fc(self):
        """Collect and prepare sets of forces and displacements.

        With probability distributions, reweighting is performed.

        This method relies on the last iteration:
        - `self.ctx.prev_nodes` is set at `run_forces_nac_calculations`.
        - `self.ctx.prev_probs` is set at `calculate_probability_distribution`.

        Reweighting factor w_N at step N is defined as

            w_N = sqrt(P(Phi_curr, u_N) / P(Phi_N, u_N))

        Phi_N: Force constants at step N.
        u_N: Displacements generated at step N.
        f_N: Forces calculated for u_N.
        P: Probability distribution for Phi_N and u_M.
        N_curr = self.ctx.iteration - 1.

        P(Phi_N, u_N) is stored in `self.ctx.prev_probs`.
        P(Phi_curr, u_N) is calculated in `collect_dataset`.
        The obtained w_N^2 is multiplied to u_N and f_N in the least squares fitting,
        i.e., the following values become the return values of this method:

        - displacements(..., N, ...) = u_N * w_N^2 in previous steps N
        - force_sets(...,N, ...) = f_N * w_N^2 in previous steps N

        Set `self.ctx.displacements` and `self.ctx.force_sets`.

        """
        self.report(f"create_dataset_for_fc_{self.ctx.iteration}")

        num_batches = self.inputs.number_of_steps_for_fitting.value

        # Initial nodes are not specified, 0K phonon is in
        # self.ctx.initial_node. This is only once used to generate random
        # displacements. In the following steps, phonons calculated at
        # specified temperature are used to generate random displacements.
        if len(self.ctx.prev_nodes) == 0:
            nodes = [
                self.ctx.initial_node,
            ]
            probs = []
        else:
            if len(self.ctx.prev_nodes) < num_batches:
                nodes = self.ctx.prev_nodes
                probs = self.ctx.prev_probs
            else:
                nodes = self.ctx.prev_nodes[-num_batches:]
                probs = self.ctx.prev_probs[-num_batches:]

        kwargs = {}
        for i, node in enumerate(nodes):
            kwargs["forces_%d" % (i + 1)] = node.outputs.force_sets
            d = get_displacements_from_phonopy_wc(node)
            kwargs["displacements_%d" % (i + 1)] = d
            if probs:
                kwargs["probability_distribution_%d" % (i + 1)] = probs[i]

        if probs:
            kwargs.update(
                {
                    "supercell": self.ctx.supercell,
                    "primitive": self.ctx.primitive,
                    "temperature": self.inputs.temperature,
                    "force_constants": self.ctx.force_constants,
                }
            )

        (displacements, force_sets, energies, included, weights) = collect_dataset(
            self.inputs.number_of_steps_for_fitting,
            self.inputs.include_ratio,
            self.inputs.linear_decay,
            **kwargs,
        )

        d, f = create_phonopy_dataset(displacements, force_sets, weights, included)
        force_sets_data = ArrayData()
        force_sets_data.set_array("force_sets", f)
        displacements_data = ArrayData()
        displacements_data.set_array("displacements", d)

        self.ctx.displacements = displacements_data
        self.ctx.force_sets = force_sets_data

        if self.ctx.force_constants is None:
            sscha_fe_data = {}
        else:
            sscha_fe_data = {"force_constants": self.ctx.force_constants}
        for i, _ in enumerate(nodes):
            sscha_fe_data[f"displacements_{i + 1}"] = kwargs[f"displacements_{i + 1}"]
        sscha_fe_data["energies"] = Dict(dict={"energies": energies})
        sscha_fe_data["included"] = Dict(dict={"included": included})
        sscha_fe_data["weights"] = Dict(dict={"weights": weights})
        calculate_sscha_free_energy(**sscha_fe_data)

    def run_force_constants_calculation_local(self):
        """Generate displacements from previous phonon calculations.

        Set `self.ctx.force_constants`.

        This method has to come after

        - `create_dataset_for_fc`

        """
        self.report("generate displacements on process")
        vals = get_force_constants_local(
            self.inputs.settings,
            self.inputs.structure,
            self.ctx.displacements,
            self.ctx.force_sets,
        )
        self.ctx.force_constants = vals["force_constants_array"]

    def run_force_constants_calculation_remote(self):
        """Run force constants calculation by PhonopyCalculation.

        This method has to come after

        - `create_dataset_for_fc`

        """
        self.report("remote force constants calculation %d" % self.ctx.iteration)

        if "code" in self.inputs:
            code = self.inputs.code
        elif "code_string" in self.inputs:
            code = Code.get_from_string(self.inputs.code_string.value)
        else:
            raise RuntimeError("code or code_string is needed.")
        builder = code.get_builder()
        builder.structure = self.inputs.structure
        builder.settings = self.inputs.settings
        builder.metadata.options.update(self.inputs.phonopy.metadata.options)
        builder.metadata.label = "Force constants calculation %d" % self.ctx.iteration
        builder.displacements = self.ctx.displacements
        builder.force_sets = self.ctx.force_sets
        future = self.submit(builder)

        self.report("Force constants remote calculation: {}".format(future.pk))
        label = "force_constants_%d" % self.ctx.iteration
        self.to_context(**{label: future})

    def set_force_constants(self):
        """Set force_constants to context for the remote fc calculation.

        Set `self.ctx.force_constants`.

        This method has to come after

        - `run_force_constants_calculation_remote`

        """
        label = "force_constants_%d" % self.ctx.iteration
        with self.ctx[label].outputs.force_constants.open(mode="rb") as source:
            with tempfile.TemporaryFile() as target:
                shutil.copyfileobj(source, target)
                target.seek(0)
                with h5py.File(target) as f:
                    self.ctx.force_constants = ArrayData()
                    self.ctx.force_constants.set_array(
                        "force_constants", f["force_constants"][:]
                    )

    def finalize(self):
        """Finalize IterHarmonicApprox."""
        self.report("IterHarmonicApprox finished at %d" % (self.ctx.iteration - 1))

    def _get_phonopy_inputs(self, displacements=None, is_initial_phonon=False):
        """Return inputs for PhonopyWorkChain."""
        inputs = {}
        inputs_orig = self.exposed_inputs(PhonopyWorkChain)
        for key in inputs_orig:
            if (
                key
                in (
                    "force_constants",
                    "displacements",
                    "displacement_dataset",
                    "force_sets",
                )
                and is_initial_phonon
            ):
                self.report(f"Set initial {key}")
                inputs[key] = inputs_orig[key]
            elif key == "calculator_inputs":
                inputs[key] = {"force": inputs_orig[key]["force"]}
                keys = list(inputs_orig[key])
                self.report(f"calculator_inputs: {keys}")
                if "nac" in inputs_orig[key]:
                    self.report(f"calculator_inputs.{key} is included.")
                    inputs[key].update({"nac": inputs_orig[key]["nac"]})
            else:
                inputs[key] = inputs_orig[key]

        if displacements is not None and not is_initial_phonon:
            inputs["displacements"] = displacements

        return inputs


def get_sscha_free_energy(
    ph: Phonopy,
    displacements,
    energies,
    weights,
    included,
    temperature,
    reference_energy=None,
):
    """Return energies for SSCHA free energy.

    SSCHA free energy is calculated from
    - Force constants (`ph.force_constants`) in eV/A^2.
    - Displacements (`displacements`) in Angstrom.
    - Supercell electronic total energy (`energies`) in eV/supercell.
    - Reference energy (`reference_energy`) in eV/supercell.
    - Reweighting values (`weights`).
    - Temperature (`temperature) in K.

    The displacements (`displacements`) are generated from this force constants
    (`ph.force_constants`). The energies (`energies`) are the results of
    some calculator (e.g. first-principels calculation) with respect to the
    displacements (`displacements`).

    Parameters
    ----------
    ph : Phonopy
        Phonopy instance ready for phonon calculation with force constants.
    displacements : list of ndarray
        Displacements of supercells.
        [(num_supercells_1, num_atoms, 3), (num_supercells_2, num_atoms, 3), ...].
    energies : list of ndarray of double
        Supercell energies.
        [(num_supercells_1,), (num_supercells_2,), ...].
    weights : list of ndarray of double
        Reweighting values calcuated by rho(Phi,disp)/rho(Phi_prev,disp).
        [(num_supercells_1,), (num_supercells_2,), ...].
    included : list of ndarray of bool
        This tells whether each supercell was included in the force constants
        calculation or not.
        [(num_supercells_1,), (num_supercells_2,), ...].
    temperature : float
        Temperature.
    reference_energy : float, optional
        Supercell energies for reference. This value is subtracted from `energies`.

    Returns
    -------
    tuple
        (harmonic free energy (F_ha)
         average energy from electric structure (<V_ave>),
         SCHA potential calculated from statistical expression (<V_ha^rho>),
         SCHA potential calculated from displacements in supercell (<V_ha^dd>))
        The energy unit is in eV/primitive-cell. In <V_ave>, `reference_energy` is
        subtracted.

    Average energy from electric structure is calcualted by A16 in Bianco
    et al., PhysRevB.96.014111.

    SSCHA free energy is given as

        F = F_ha - <V_ha> + <V_ave>.

    """
    from phono3py.sscha.sscha import SupercellPhonon, DispCorrMatrix

    if reference_energy is None:
        ref_e = 0
    else:
        ref_e = reference_energy

    # <V_ha> with correlation of displacements analytically calcualted from density
    # matrix.
    sc_ph = SupercellPhonon(ph.supercell, ph.force_constants)
    cutoff_frequency = ph.thermal_properties.cutoff_frequency / THzToEv
    uu = DispCorrMatrix(sc_ph, cutoff_frequency=cutoff_frequency)
    uu.run(temperature)
    v_harm_rho = (sc_ph.force_constants * uu.psi_matrix).sum() / 2

    v_harm_dd = 0.0
    v_ave = 0.0
    for d, e, w, inc in zip(displacements, energies, weights, included):
        # <V_ha> from correlation of generated finite displacements.
        _w = np.extract(inc, w)
        _e = np.extract(inc, e)
        u = d[inc].reshape(d[inc].shape[0], -1)
        # uu_ave = np.zeros_like(sc_ph.force_constants)
        # for u_sp, w_sp in zip(u, _w):
        #     uu_ave += np.outer(u_sp, u_sp) * w_sp
        # uu_ave is written in compact form as below:
        uu_ave = np.dot(u.T * _w, u)
        v_harm_dd += (sc_ph.force_constants * uu_ave).sum() / 2

        # <V_el>
        v_ave += np.sum((_e - ref_e) * _w)

    n_div = np.concatenate(included).sum()
    v_harm_dd /= n_div
    v_ave /= n_div

    # F_ha
    free_energy = ph.get_thermal_properties_dict()["free_energy"][0]
    free_energy /= EvTokJmol  # in eV/primitive cell

    # fe_corr in eV/primitive cell
    if ph.primitive_matrix is None:
        det_pmat = 1
    else:
        det_pmat = np.linalg.det(ph.primitive_matrix)
    coef_per_prim = det_pmat / np.linalg.det(ph.supercell_matrix)
    v_ave *= coef_per_prim
    v_harm_rho *= coef_per_prim
    v_harm_dd *= coef_per_prim

    return free_energy, v_ave, v_harm_rho, v_harm_dd


def collect_dataset(number_of_steps_for_fitting, include_ratio, linear_decay, **data):
    """Collect supercell displacements, forces, and energies.

    Returns
    -------
    See docstring of `get_sshca_free_energy_dataset`.

    """
    nitems = max([int(key.split("_")[-1]) for key in data.keys() if "forces" in key])

    force_sets_in_db = []
    displacements_in_db = []
    probs_in_db = []
    for i in range(nitems):
        force_sets_in_db.append(data[f"forces_{i + 1}"])
        displacements_in_db.append(data[f"displacements_{i + 1}"])
        if f"probability_distribution_{i + 1}" in data:
            probs_in_db.append(data[f"probability_distribution_{i + 1}"])

    kwargs = {
        "max_items": number_of_steps_for_fitting.value,
        "ratio": include_ratio.value,
        "linear_decay": linear_decay.value,
    }
    if probs_in_db:
        kwargs["probs_in_db"] = probs_in_db
    if "force_constants" in data:
        kwargs["force_constants"] = data["force_constants"].get_array("force_constants")
    if "temperature" in data:
        kwargs["temperature"] = data["temperature"].value
    if "supercell" in data:
        kwargs["supercell"] = phonopy_atoms_from_structure(data["supercell"])
    if "primitive" in data:
        kwargs["primitive"] = phonopy_atoms_from_structure(data["primitive"])

    (
        displacements,
        force_sets,
        energies,
        included,
        weights,
    ) = get_sshca_free_energy_dataset(displacements_in_db, force_sets_in_db, **kwargs)

    return displacements, force_sets, energies, included, weights


@calcfunction
def calculate_sscha_free_energy(**data):
    """Calculate SSCHA free energy.

    To be implemented. Currently this function is used for storing data in DB.

    """
    pass


def create_phonopy_dataset(displacements, force_sets, weights, included):
    """Create dataset for phonopy force constants calculation.

    This function does two jobs:
    - Reweights forces and displacements.
    - Concatenate data of selected supercells.

    Parameters
    ----------
    displacements : list
        List of sets of supercell displacements included.
        shape=(num_steps, num_supercell, natom_supercell, 3), dtype='double'
    force_sets : list
        List of sets of supercell forces included.
        shape=(num_steps, num_supercell, natom_supercell, 3), dtype='double'
    weights : list of ndarray of double
        Weights multiplied to forces and displacements.
        Same shape as `included`.
    included : list of list of bool
        This gives information that each supercell is included or not.
        Maybe `len(d) != len(np.concatenate(included)) , but
        `len(d) == np.concatenate(included).sum()`.

    Returns
    -------
    d : ndarray
        Sets of supercell displacements included.
        shape=(num_included, natom_supercell, 3), dtype='double'
    f : ndarray
        Sets of supercell forces included.
        shape=(num_included, natom_supercell, 3), dtype='double'

    """
    _reweight_dataset(displacements, force_sets, weights)
    d, f = _concatenate_dataset(displacements, force_sets, included)
    return d, f


def get_sshca_free_energy_dataset(
    displacements_in_db,
    force_sets_in_db,
    probs_in_db=None,
    max_items=None,
    ratio=None,
    linear_decay=False,
    force_constants=None,
    temperature=None,
    supercell=None,
    primitive=None,
):
    """Return data needed to calculated SSHCA free energy.

    Parameters
    ----------
    force_sets_in_db : List of ArrayData
        Each ArrayData is the output of PhonopyWorkChain, i.e.,
            [(n_snapshots, n_satom, 3), (n_snapshots, n_satom, 3), ...]
    displacements_in_db : List of ArrayData
        Each ArrayData is the output of PhonopyWorkChain, i.e.,
            [(n_snapshots, n_satom, 3), (n_snapshots, n_satom, 3), ...]
    probs_in_db : List of ArrayData, optional
        Probability distributions of corresponding displacements and force constants.
            [(n_snapshots,), (n_snapshots,), ...]
        If this is given, forces and displacements are reweighted. To reweight,
        force_constants has to be given.
    force_constants : ArrayData, optional
        Force constants. Used for reweighting.
    temperature : float, optional
        Temperature. Used for reweighting.
    supercell : PhonopyAtoms, optional
        Supercell. Used for reweighting.
    primitive : PhonopyAtoms, optional
        Primitive cell. Used for reweighting.

    Returns
    -------
    displacements : list
        List of sets of supercell displacements included.
        shape=(num_steps, num_supercell, natom_supercell, 3), dtype='double'
    force_sets : list
        List of sets of supercell forces included.
        shape=(num_steps, num_supercell, natom_supercell, 3), dtype='double'
    included : list of list of bool
        This gives information that each supercell is included or not.
        Maybe `len(d) != len(np.concatenate(included)) , but
        `len(d) == np.concatenate(included).sum()`.
    energies : list of ndarray of double
        List of supercell energies in all PhonopyWorkChains calculations.
        Same shape as `included`.
    weights : list of ndarray of double
        Weights multiplied to forces and displacements.
        Same shape as `included`.

    """
    displacements, force_sets, energies, probs = _extract_dataset_from_db(
        force_sets_in_db, displacements_in_db, probs_in_db=probs_in_db
    )
    num_elems = [len(d_batch) for d_batch in displacements]
    included = _select_snapshots(
        num_elems,
        energies,
        max_items=max_items,
        ratio=ratio,
        linear_decay=linear_decay,
    )
    if probs:
        weights = _get_reweights(
            probs, force_constants, displacements, temperature, supercell, primitive
        )
    else:
        weights = [np.ones(len(disps), dtype="double") for disps in displacements]

    return displacements, force_sets, energies, included, weights


def _extract_dataset_from_db(force_sets_in_db, displacements_in_db, probs_in_db=None):
    """Collect force_sets, energies, and displacements to numpy arrays.

    Parameters
    ----------
    force_sets_in_db : List of ArrayData
        Each ArrayData is the output of PhonopyWorkChain.
    displacements_in_db : List of ArrayData
        Each ArrayData is the output of PhonopyWorkChain.
    probs_in_db : List of ArrayData or None
        Probability distributions of corresponding displacements and force constants.

    Returns
    -------
    my_force_sets : List of ndarray
    my_energies : List of ndarray
    my_displacements : List of ndarray

    """
    nitems = len(force_sets_in_db)
    my_displacements = []
    my_force_sets = []
    my_energies = []
    my_probs = []

    for i in range(nitems):
        force_sets = force_sets_in_db[i].get_array("force_sets")
        displacements = displacements_in_db[i].get_array("displacements")

        my_force_sets.append(force_sets)
        if "energies" in force_sets_in_db[i].get_arraynames():
            energy_sets = force_sets_in_db[i].get_array("energies")
            my_energies.append(energy_sets)
        my_displacements.append(displacements)

        if probs_in_db:
            my_probs.append(probs_in_db[i].get_array("probability_distributions"))

    return my_displacements, my_force_sets, my_energies, my_probs


def _select_snapshots(
    num_elems,
    energies,
    max_items=None,
    ratio=None,
    linear_decay=False,
):
    """Select snapshots.

    Returns
    -------
    included : list of list of bool
        This gives information that each supercell is included or not.
        Maybe `len(d) != len(np.concatenate(included)) , but
        `len(d) == np.concatenate(included).sum()`.

    """
    included = _choose_snapshots_by_linear_decay(
        num_elems, max_items=max_items, linear_decay=linear_decay
    )

    # Remove snapshots that have high energies when include_ratio is given.
    if energies is not None and ratio is not None:
        if 0 < ratio and ratio < 1:
            included = _remove_high_energy_snapshots(energies, included, ratio)

    return included


def _get_reweights(
    probs,
    force_constants,
    displacements,
    temperature: float,
    supercell: PhonopyAtoms,
    primitive: PhonopyAtoms,
):
    assert force_constants is not None
    assert temperature is not None
    assert supercell is not None
    assert primitive is not None
    assert len(probs) == len(displacements)
    _primitive = convert_to_phonopy_primitive(supercell, primitive)
    if force_constants.shape[0] == force_constants.shape[1]:
        fc = force_constants
    else:
        fc = _compact_fc_to_full_fc(supercell, _primitive, force_constants)
    weights = []
    for disps, prob_orig in zip(displacements, probs):
        prob = get_probability_distribution(
            supercell, _primitive, fc, disps, temperature
        )
        weights.append(prob / prob_orig)
    return weights


def _reweight_dataset(displacements, force_sets, weights):
    """Apply reweighting when the value is smaller than 1.

    Note
    ----
    `displacements` and `force_sets` are overwritten.

    """
    for disps, forces, weights_at_batch in zip(displacements, force_sets, weights):
        for i, w in enumerate(weights_at_batch):
            if w < 1:
                disps[i] *= w
                forces[i] *= w


def _concatenate_dataset(displacements, force_sets, included):
    _f = [force_sets[i][included_batch] for i, included_batch in enumerate(included)]
    _d = [displacements[i][included_batch] for i, included_batch in enumerate(included)]
    return np.concatenate(_d, axis=0), np.concatenate(_f, axis=0)


def _choose_snapshots_by_linear_decay(num_elems, max_items=None, linear_decay=False):
    """Choose snapshots by linear_decay.

    With linear_decay=True, numbers of snapshots to be taken
    are biased. Older snapshots are taken lesser. The fraction
    of the number of snapshots in each previous phonon calculation
    (many snapshots in one phonon calculation) to be taken is defined
    linearly by

        ratios = (np.arange(max_items, dtype=float) + 1) / max_items

    where max_items is the number of previous phonon calculations to be
    included at maximum.

    """
    nitems = len(num_elems)
    if max_items is None:
        _max_items = nitems
    else:
        _max_items = max_items

    if linear_decay:
        ratios = (np.arange(_max_items, dtype=float) + 1) / _max_items
    else:
        ratios = np.ones(_max_items, dtype=int)
    ratios = ratios[-nitems:]
    included = []

    for i, n in enumerate(num_elems):
        n_in = int(ratios[i] * n + 0.5)
        if n < n_in:
            n_in = n
        included.append(
            [
                True,
            ]
            * n_in
            + [
                False,
            ]
            * (n - n_in)
        )

    return included


def _remove_high_energy_snapshots(energies, included, ratio):
    """Reject high energy snapshots.

    Parameters
    ----------
    energies : list of ndarray
        List of supercell total energies in each batch
    included : list of list of bool
        List of list of True/False as included snapshots in each batch
        Rejected elements are turned to False.
    ratio : float
        How much ratio of lowest energy snapshots is included after
        sorting by energy.

    Returns
    -------
    ret_included :
        Rejected snapshots are turned to False from 'included'.

    """
    concat_included = np.concatenate(included)
    concat_energies = np.concatenate(energies)
    included_energies = concat_energies[concat_included]
    included_indices = np.arange(len(concat_included))[concat_included]

    num_include = int(ratio * len(included_energies) + 0.5)
    if len(included_energies) < num_include:
        num_include = len(included_energies)
    _indices = np.argsort(included_energies)[:num_include]
    included_indices_after_energy = included_indices[_indices]

    bool_list = [
        False,
    ] * len(concat_included)
    for i in included_indices_after_energy:
        bool_list[i] = True
    ret_included = []
    count = 0
    for included_batch in included:
        ret_included.append(bool_list[count : (count + len(included_batch))])
        count += len(included_batch)
    return ret_included


@calcfunction
def get_probability_distribution_data(
    supercell,
    primitive,
    force_constants,
    random_displacements,
    temperature,
):
    """Calculate probability distributions for sets of random displacements."""
    phonopy_atoms_supercell = phonopy_atoms_from_structure(supercell)
    phonopy_atoms_primitive = phonopy_atoms_from_structure(primitive)
    phonopy_primitive = convert_to_phonopy_primitive(
        phonopy_atoms_supercell,
        phonopy_atoms_primitive,
    )
    _force_constants = force_constants.get_array("force_constants")
    _prob = get_probability_distribution(
        phonopy_atoms_supercell,
        phonopy_primitive,
        _force_constants,
        random_displacements.get_array("displacements"),
        temperature.value,
    )
    prob = ArrayData()
    prob.set_array("probability_distributions", _prob)
    return {"probability_distributions": prob}


def get_probability_distribution(
    supercell: PhonopyAtoms,
    primitive: Primitive,
    force_constants,
    random_displacements,
    temperature,
):
    """Calculate probability distributions for sets of random displacements."""
    from phono3py.sscha.sscha import get_sscha_matrices

    if force_constants.shape[0] == force_constants.shape[1]:
        fc = force_constants
    else:
        fc = _compact_fc_to_full_fc(supercell, primitive, force_constants)
    uu = get_sscha_matrices(supercell, fc)
    uu.run(temperature)
    dmat = random_displacements.reshape(-1, 3 * len(supercell))
    vals = -(dmat * np.dot(dmat, uu.upsilon_matrix)).sum(axis=1) / 2
    return uu.prefactor * np.exp(vals)


def _compact_fc_to_full_fc(
    supercell: PhonopyAtoms, primitive: Primitive, force_constants
):
    fc = np.zeros(
        (force_constants.shape[1], force_constants.shape[1], 3, 3),
        dtype="double",
        order="C",
    )
    fc[primitive.p2s_map] = force_constants
    distribute_force_constants_by_translations(fc, primitive, supercell)
    return fc


@calcfunction
def get_force_constants_local(settings, structure, displacements, force_sets):
    """Calcfunction to store force constants data."""
    smat = settings["supercell_matrix"]
    ph = Phonopy(
        phonopy_atoms_from_structure(structure),
        supercell_matrix=smat,
        primitive_matrix="auto",
    )
    d = displacements.get_array("displacements")
    f = force_sets.get_array("force_sets")
    ph.dataset = {"displacements": d, "forces": f}
    ph.produce_force_constants(fc_calculator="alm")
    force_constants_array = ArrayData()
    force_constants_array.set_array("force_constants", ph.force_constants)

    with tempfile.TemporaryDirectory() as dname:
        filename = os.path.join(dname, "force_constants.hdf5")
        write_force_constants_to_hdf5(
            ph.force_constants,
            filename=filename,
            p2s_map=ph.primitive.p2s_map,
        )
        force_constants_file = SinglefileData(
            file=filename, filename="force_constants.hdf5"
        )

    return {
        "force_constants_array": force_constants_array,
        "force_constants_file": force_constants_file,
    }
