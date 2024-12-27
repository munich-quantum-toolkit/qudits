from __future__ import annotations

from typing import TYPE_CHECKING

from .jobstatus import JobStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..backends.backendv2 import Backend
    from . import JobResult
    from .client_api import APIClient


class Job:
    def __init__(self, backend: Backend, job_id: str = "local_sim", api_client: APIClient | None = None) -> None:
        self._backend = backend
        self._job_id = job_id
        self._api_client = api_client
        self.set_status(JobStatus.INITIALIZING)
        self._result: JobResult | None = None

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def backend(self) -> Backend:
        return self._backend

    def status(self) -> JobStatus:
        if self._api_client:
            self.set_status(self._api_client.get_job_status(self._job_id))
        else:
            # For local simulation, we assume the job is done immediately
            self.set_status(JobStatus.DONE)
        return self._status

    def result(self) -> JobResult:
        cached_result = self._result
        if cached_result is not None:
            return cached_result

        self._wait_for_final_state()

        if self._api_client:
            self._result = self._api_client.get_job_result(self._job_id)
        else:
            msg = "If the job is not run on the machine, then the result should be given by the simulation already. "
            raise RuntimeError(msg)

        return self._result

    def _wait_for_final_state(self, callback: Callable[[str, JobStatus], None] | None = None) -> None:
        if self._api_client:
            try:
                # Using a synchronous wait implementation
                self._api_client.wait_for_job_completion(self._job_id, callback)
            except Exception as e:
                msg = f"Error while waiting for job {self._job_id}: {e!s}"
                raise RuntimeError(msg) from e
        else:
            # For local simulation, we assume the job is done immediately
            self.set_status(JobStatus.DONE)
            if callback:
                callback(self._job_id, self._status)

    def cancelled(self) -> bool:
        return self._status == JobStatus.CANCELLED

    def done(self) -> bool:
        return self._status == JobStatus.DONE

    def running(self) -> bool:
        return self._status == JobStatus.RUNNING

    def in_final_state(self) -> bool:
        return self._status.is_final

    def set_result(self, result: JobResult) -> None:
        self._result = result

    def set_status(self, new_status: JobStatus) -> None:
        self._status = new_status
