from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING, cast

import requests

from mqt.qudits.simulation.jobs import JobResult
from mqt.qudits.simulation.jobs.config_api import (
    BASE_URL,
    JOB_RESULT_ENDPOINT,
    JOB_STATUS_ENDPOINT,
    SUBMIT_JOB_ENDPOINT,
)
from mqt.qudits.simulation.jobs.jobstatus import JobStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from mqt.qudits.core import LevelGraph
    from mqt.qudits.quantum_circuit import QuantumCircuit


class APIClient:
    def __init__(self) -> None:
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def submit_job(self, circuit: QuantumCircuit, shots: int, energy_level_graphs: list[LevelGraph]) -> str:
        url = f"{BASE_URL}{SUBMIT_JOB_ENDPOINT}"
        payload = {
            "circuit": circuit.to_qasm(),
            "shots": shots,
            "energy_level_graphs": list(energy_level_graphs),
        }
        response = self.session.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return cast("str", data.get("job_id"))
        msg = f"Job submission failed with status code {response.status_code}"
        raise RuntimeError(msg)

    def get_job_status(self, job_id: str) -> JobStatus:
        url = f"{BASE_URL}{JOB_STATUS_ENDPOINT}/{job_id}"
        response = self.session.get(url)
        if response.status_code == 200:
            data = response.json()
            return JobStatus.from_string(data.get("status"))
        msg = f"Failed to get job status with status code {response.status_code}"
        raise RuntimeError(msg)

    def get_job_result(self, job_id: str) -> JobResult:
        url = f"{BASE_URL}{JOB_RESULT_ENDPOINT}/{job_id}"
        response = self.session.get(url)
        if response.status_code == 200:
            data = response.json()
            return JobResult(data)
        msg = f"Failed to get job result with status code {response.status_code}"
        raise RuntimeError(msg)

    def wait_for_job_completion(
        self,
        job_id: str,
        callback: Callable[[str, JobStatus], None] | None = None,
        polling_interval: float = 5,
    ) -> JobStatus:
        while True:
            status = self.get_job_status(job_id)
            if callback:
                callback(job_id, status)
            if status.is_final:
                return status
            sleep(polling_interval)
