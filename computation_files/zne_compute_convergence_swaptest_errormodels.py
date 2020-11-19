from qiskit import *
from qiskit.providers.aer.noise import *
from qiskit.test.mock import FakeVigo
import numpy as np

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)

from zne_circuits import qc_swaptest, swaptest_exp_val_func
from zero_noise_extrapolation_cnot import *

def split_noise_model(noise_model):
    noise_dict = noise_model.to_dict()

    cnot_errors = {"errors": []}
    singleq_errors = {"errors": []}
    measurement_errors = {"errors": []}

    for err in noise_dict["errors"]:
        if err["type"] == "qerror":
            if "cx" in err["operations"]:
                cnot_errors["errors"].append(err)
            else:
                singleq_errors["errors"].append(err)
        elif err["type"] == "roerror":
            measurement_errors["errors"].append(err)
        else:
            print(err["type"], "not recognised")
    cnot_and_meas_errors = {"errors": cnot_errors["errors"] + measurement_errors["errors"]}
    cnot_and_singleq_errors = {"errors": cnot_errors["errors"] + singleq_errors["errors"]}

    return NoiseModel.from_dict(cnot_errors), NoiseModel.from_dict(cnot_and_meas_errors), \
           NoiseModel.from_dict(cnot_and_singleq_errors)

if __name__ == "__main__":
    FILENAME = abs_path + "/data_files" + "/zne_errormodels_mockbackend.npz"

    FILENAME_OBJ = abs_path + "/data_files" + "/zne_convergence_mockbackend_obj.npz"

    file_obj = np.load(FILENAME_OBJ, allow_pickle=True)
    qem = file_obj["qem"][()]

    shots = 8192

    amp_factors = 10

    mock_backend = FakeVigo()
    sim_backend = Aer.get_backend("qasm_simulator")

    noise_model = NoiseModel.from_backend(mock_backend)
    #print(noise_model.to_dict())

    cnot_noise, cnotandmeas_noise, cnotandsingleq_noise = split_noise_model(noise_model)

    #print(cnot_noise.to_dict())
    #print(cnotandmeas_noise.to_dict())
    #print(cnotandsingleq_noise.to_dict())

    qem_onlycnot_noise = ZeroNoiseExtrapolation(qc_swaptest, exp_val_func=swaptest_exp_val_func, shots=shots,
                                backend=sim_backend, noise_model=cnot_noise, n_amp_factors=amp_factors)
    qem_cnotandmeas_noise = ZeroNoiseExtrapolation(qc_swaptest, exp_val_func=swaptest_exp_val_func, shots=shots,
                                backend=sim_backend, noise_model=cnotandmeas_noise, n_amp_factors=amp_factors)
    qem_cnotandsingleq_noise = ZeroNoiseExtrapolation(qc_swaptest, exp_val_func=swaptest_exp_val_func, shots=shots,
                                backend=sim_backend, noise_model=cnotandsingleq_noise, n_amp_factors=amp_factors)

    # Use the circuit as transpiled for the FakeVigo-backend
    qem_onlycnot_noise.qc = qem.qc
    qem_cnotandmeas_noise.qc = qem.qc
    qem_cnotandsingleq_noise.qc = qem.qc

    repeats = 5000

    qem_onlycnot_noise.mitigate(repeats=repeats, verbose=True)
    qem_cnotandmeas_noise.mitigate(repeats=repeats, verbose=True)
    qem_cnotandsingleq_noise.mitigate(repeats=repeats, verbose=True)

    mitigated_onlycnot = zeros(amp_factors)
    mitigated_cnotandmeas = zeros(amp_factors)
    mitigated_cnotandsingleq = zeros(amp_factors)

    mitigated_onlycnot[0] = np.average(qem_onlycnot_noise.bare_exp_vals)
    mitigated_cnotandmeas[0] = np.average(qem_cnotandmeas_noise.bare_exp_vals)
    mitigated_cnotandsingleq[0] = np.average(qem_cnotandsingleq_noise.bare_exp_vals)

    amplification_factors = np.asarray([1+2*i for i in range(amp_factors)])

    for i in range(1, amp_factors):
        mitigated_onlycnot[i] = richardson_extrapolate(
            np.average(qem_onlycnot_noise.all_exp_vals[:,0:i+1], axis=0), amplification_factors[0:i+1])
        mitigated_cnotandmeas[i] = richardson_extrapolate(
            np.average(qem_cnotandmeas_noise.all_exp_vals[:, 0:i + 1], axis=0), amplification_factors[0:i + 1])
        mitigated_cnotandsingleq[i] = richardson_extrapolate(
            np.average(qem_cnotandsingleq_noise.all_exp_vals[:, 0:i + 1], axis=0), amplification_factors[0:i + 1])

    print(mitigated_onlycnot)
    print(mitigated_cnotandmeas)
    print(mitigated_cnotandsingleq)

    np.savez(FILENAME, backend_name=mock_backend.name(), mitigated_onlycnot=mitigated_onlycnot,
             mitigated_cnotandmeas=mitigated_cnotandmeas, mitigated_cnotandsingleq=mitigated_cnotandsingleq,
             cnot_noise_model=cnot_noise.to_dict(), cnotandmeas_noise_model=cnotandmeas_noise.to_dict(),
             cnotandsingleq_noise_model=cnotandsingleq_noise.to_dict(), amp_factors=amp_factors,
             shots=shots, repeats=repeats, tot_shots=(shots*repeats))