# quantum_error_mitigation

This repository contains the code developed for my specialization project in physics (15 ETCS credits), as a part of my Master's degree in Applied physics at NTNU. The project focuses on Quantum Error Mitigation (QEM) for near term quantum computers, and will serve as a preparation to my master thesis on the same subject.

The main implementation is that of the zero-noise extrapolation QEM-technique for mitigating noise in CNOT-gates, with a novel method of noise amplification. We amplify general noise in CNOT-gates by repeating all CNOT-gates in the circuit, taking advantage of the fact that the CNOT-gate is its own inverse. A general implementation of this is found in the zero_noise_extrapolation_cnot.py script. An updated version of this has been made a part of the OpenQuantumComputation project, lead by my main supervisor Franz G. Fuchs, and can be found in the https://github.com/OpenQuantumComputing/error_mitigation repository.

# Acknowlegdements

Thank you to my main supervisor Franz G. Fuchs for help and support throughout the project, and my co-supervisors Jeroen Danon and Knut-Andreas Lie for good guidance and thorough proof reading.
