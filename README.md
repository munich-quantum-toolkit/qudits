[![PyPI](https://img.shields.io/pypi/v/mqt.qudits?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.qudits/)
![OS](https://img.shields.io/badge/os-linux%20%7C%20macos%20%7C%20windows-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/qudits/ci.yml?branch=main&style=flat-square&logo=github&label=ci)](https://github.com/munich-quantum-toolkit/qudits/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/qudits/cd.yml?style=flat-square&logo=github&label=cd)](https://github.com/munich-quantum-toolkit/qudits/actions/workflows/cd.yml)
[![Documentation](https://img.shields.io/readthedocs/mqt-qudits?logo=readthedocs&style=flat-square)](https://mqt.readthedocs.io/projects/qudits)
[![codecov](https://img.shields.io/codecov/c/github/munich-quantum-toolkit/qudits?style=flat-square&logo=codecov)](https://codecov.io/gh/munich-quantum-toolkit/qudits)

> [!NOTE]
> This project is currently in low maintenance mode.
> We will still fix bugs and accept pull requests, but we will not actively develop new features.

# MQT Qudits - A Framework For Mixed-Dimensional Qudit Quantum Computing

MQT Qudits is a framework for research and education for mixed-dimensional qudit quantum computing.
It is part of the [_Munich Quantum Toolkit (MQT)_](https://mqt.readthedocs.io).

<p align="center">
  <a href="https://mqt.readthedocs.io/projects/qudits">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

## Key features

- **Mixed-dimensional quantum circuit support**: Design, simulate, and analyze quantum circuits with arbitrary qudit dimensions, not limited to qubits. [Tutorial](https://mqt.readthedocs.io/projects/qudits/en/latest/tutorial.html)
- **Python-first API**: Intuitive Python interface for circuit construction, simulation, and analysis, with type hints and integration with scientific Python libraries.
- **Cross-platform and open-source**: C++20 core with Python bindings, prebuilt wheels for Linux, macOS, and Windows via [PyPI](https://pypi.org/project/mqt.qudits/).

If you have any questions, feel free to create a [discussion](https://github.com/munich-quantum-toolkit/qudits/discussions) or an [issue](https://github.com/munich-quantum-toolkit/qudits/issues) on [GitHub](https://github.com/munich-quantum-toolkit/qudits).

## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Partner Logos">
  </picture>
</p>

Thank you to all the contributors who have helped make MQT Qudits a reality!

<p align="center">
  <a href="https://github.com/munich-quantum-toolkit/qudits/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/qudits" alt="Contributors to munich-quantum-toolkit/qudits" />
  </a>
</p>

The MQT will remain free, open-source, and permissively licensed—now and in the future.
We are firmly committed to keeping it open and actively maintained for the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories: https://github.com/munich-quantum-toolkit
- Contributing code, documentation, tests, or examples via issues and pull requests
- Citing the MQT in your publications (see [Cite This](#cite-this))
- Citing our research in your publications (see [References](https://mqt.readthedocs.io/projects/qudits/en/latest/references.html))
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: https://github.com/sponsors/munich-quantum-toolkit

<p align="center">
  <a href="https://github.com/sponsors/munich-quantum-toolkit">
  <img width=20% src="https://img.shields.io/badge/Sponsor-white?style=for-the-badge&logo=githubsponsors&labelColor=black&color=blue" alt="Sponsor the MQT" />
  </a>
</p>

## Getting Started

`mqt.qudits` is available via [PyPI](https://pypi.org/project/mqt.qudits/).

```console
(.venv) $ pip install mqt.qudits
```

**Detailed documentation and examples are available at [ReadTheDocs](https://mqt.readthedocs.io/projects/qudits).**

> [!NOTE]
> Some of the functionalities of MQT Qudits are illustrated in [this video](https://www.youtube.com/watch?v=due_CX7H85A).

## System Requirements and Building

Building the project requires a C++ compiler with support for C++20 and CMake 3.24 or newer.
For details on how to build the project, please refer to the [documentation](https://mqt.readthedocs.io/projects/qudits).
Building (and running) is continuously tested under Linux, macOS, and Windows using the [latest available system versions for GitHub Actions](https://github.com/actions/runner-images).
MQT Qudits is compatible with all [officially supported Python versions](https://devguide.python.org/versions/).

## Cite This

Please cite the work that best fits your use case.

### MQT Qudits (the tool)

When citing the software itself or results produced with it, cite the MQT Qudits paper:

```bibtex
@misc{mato2024mqtquditssoftwareframework,
  title        = {{{MQT Qudits}}: {{A}} Software Framework for Mixed-Dimensional Quantum Computing},
  author       = {Mato, Kevin and Ringbauer, Martin and Burgholzer, Lukas and Wille, Robert},
  year         = {2024},
  url          = {https://arxiv.org/abs/2410.02854},
  eprint       = {2410.02854},
  eprinttype   = {arXiv}
}
```

### The Munich Quantum Toolkit (the project)

When discussing the overall MQT project or its ecosystem, cite the MQT Handbook:

```bibtex
@inproceedings{mqt,
  title        = {The {{MQT}} Handbook: {{A}} Summary of Design Automation Tools and Software for Quantum Computing},
  shorttitle   = {{The MQT Handbook}},
  author       = {Wille, Robert and Berent, Lucas and Forster, Tobias and Kunasaikaran, Jagatheesan and Mato, Kevin and Peham, Tom and Quetschlich, Nils and Rovara, Damian and Sander, Aaron and Schmid, Ludwig and Schoenberger, Daniel and Stade, Yannick and Burgholzer, Lukas},
  year         = 2024,
  booktitle    = {IEEE International Conference on Quantum Software (QSW)},
  doi          = {10.1109/QSW62656.2024.00013},
  eprint       = {2405.17543},
  eprinttype   = {arxiv},
  addendum     = {A live version of this document is available at \url{https://mqt.readthedocs.io}}
}
```

### Peer-Reviewed Research

When citing the underlying methods and research, please reference the most relevant peer-reviewed publications from the list below:

[[1]](https://arxiv.org/pdf/2410.02854)
K. Mato, M. Ringbauer, L. Burgholzer, R. Wille.
MQT Qudits: A Software Framework for Mixed-Dimensional Quantum Computing

[[2]](https://www.cda.cit.tum.de/files/eda/2024_dac_mixed_dimensional_qudit_state_preparation_using_edge_weighted_decision_diagrams.pdf)
K. Mato, S. Hillmich, and R. Wille.
Mixed-Dimensional Qudit State Preparation Using Edge-Weighted Decision Diagrams.
_Design Automation Conference (DAC)_, 2024

[[3]](https://www.cda.cit.tum.de/files/eda/2023_qce_mixed_dimensional_quantum_circuit_simulation_with_decision_diagrams.pdf)
K. Mato, S. Hillmich, and R. Wille.
Mixed-Dimensional Quantum Circuit Simulation with Decision Diagrams.
_International Conference on Quantum Computing and Engineering (QCE)_, 2023.

[[4]](https://www.cda.cit.tum.de/files/eda/2023_qsw_compression_of_qubit_circuits.pdf)
K. Mato, S. Hillmich, and R. Wille.
Compression of Qubit Circuits: Mapping to Mixed-Dimensional Quantum Systems.
_International Conference on Quantum Software (QSW)_, 2023.

[[5]](https://www.cda.cit.tum.de/files/eda/2023_aspdac_qudit_entanglement_compilation.pdf)
K. Mato, M. Ringbauer, S. Hillmich, and R. Wille.
Compilation of Entangling Gates for High-Dimensional Quantum Systems.
_Asia and South Pacific Design Automation Conference (ASP-DAC)_, 2023.

[[6]](https://www.cda.cit.tum.de/files/eda/2022_qce_adaptive_compilation_of_multi_level_quantum_operations.pdf)
K. Mato, M. Ringbauer, S. Hillmich, and R. Wille.
Adaptive Compilation of Multi-Level Quantum Operations.
_International Conference on Quantum Computing and Engineering (QCE)_, 2022.

---

## Acknowledgements

MQT Qudits is the result of the project NeQST funded by the European Union under the Horizon Europe program (grant agreement No. 101080086).
However, views and opinions expressed are those of the author(s) only and do not necessarily reflect those of the European Union or the European Commission.
Neither the European Union nor the granting authority can be held responsible for them.

The Munich Quantum Toolkit has been supported by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 101001318), the Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, the Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus, as well as the Unitary Foundation.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>
