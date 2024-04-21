from __future__ import annotations

import os
import time
from typing import Callable, NoReturn

from ...exceptions import JobError, JobTimeoutError
from .jobstatus import JobStatus


class Job:
    """Class to handle jobs

    This first version of the Backend abstract class is written to be mostly
    backwards compatible with the legacy providers interface. This was done to ease
    the transition for users and provider maintainers to the new versioned providers.
    Expect future versions of this abstract class to change the data model and
    interface.
    """

    version = 1
    _async = True

    def __init__(self, backend: Backend | None, job_id: str = "auto", **kwargs) -> None:
        """Initializes the asynchronous job.

        Args:
            backend: the backend used to run the job.
            job_id: a unique id in the context of the backend used to run the job.
            kwargs: Any key-value metadata to associate with this job.
        """
        if job_id == "auto":
            current_time = int(time.time() * 1000)
            self._job_id = hash((os.getpid(), current_time))
        else:
            self._job_id = job_id
        self._backend = backend
        self.metadata = kwargs

    def job_id(self) -> str:
        """Return a unique id identifying the job."""
        return self._job_id

    def backend(self) -> Backend:
        """Return the backend where this job was executed."""
        if self._backend is None:
            msg = "The job does not have any backend."
            raise JobError(msg)
        return self._backend

    def done(self) -> bool:
        """Return whether the job has successfully run."""
        return self.status() == JobStatus.DONE

    def running(self) -> bool:
        """Return whether the job is actively running."""
        return self.status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        """Return whether the job has been cancelled."""
        return self.status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        """Return whether the job is in a final job state such as DONE or ERROR."""
        return self.status() in JobStatus.JOB_FINAL_STATES

    def wait_for_final_state(
        self, timeout: float | None = None, wait: float = 5, callback: Callable | None = None
    ) -> None:
        """Poll the job status until it progresses to a final state such as DONE or ERROR.

        Args:
            timeout: Seconds to wait for the job. If None, wait indefinitely.
            wait: Seconds between queries.
            callback: Callback function invoked after each query.

        Raises:
            JobTimeoutError: If the job does not reach a final state before the specified timeout.
        """
        if not self._async:
            return
        start_time = time.time()
        status = self.status()
        while status not in JobStatus.JOB_FINAL_STATES:
            elapsed_time = time.time() - start_time
            if timeout is not None and elapsed_time >= timeout:
                msg = f"Timeout while waiting for job {self.job_id()}."
                raise JobTimeoutError(msg)
            if callback:
                callback(self.job_id(), status, self)
            time.sleep(wait)
            status = self.status()

    def submit(self) -> NoReturn:
        """Submit the job to the backend for execution."""
        raise NotImplementedError

    def result(self):
        """Return the results of the job."""
        return self._result

    def set_result(self, result) -> None:
        self._result = result

    def cancel(self) -> NoReturn:
        """Attempt to cancel the job."""
        raise NotImplementedError

    def status(self) -> str:
        """Return the status of the job, among the values of BackendStatus."""
        raise NotImplementedError
