from __future__ import annotations

import enum


class JobStatus(str, enum.Enum):
    """Enumeration for job status."""

    INITIALIZING = "initializing"
    QUEUED = "queued"
    VALIDATING = "validating"
    RUNNING = "running"
    CANCELLED = "cancelled"
    DONE = "done"
    ERROR = "error"

    def __str__(self) -> str:
        """Return a human-readable description of the status."""
        return self.description

    @property
    def description(self) -> str:
        """Return a detailed description of the status."""
        return {
            self.INITIALIZING: "Job is being initialized",
            self.QUEUED: "Job is waiting in the queue",
            self.VALIDATING: "Job is being validated",
            self.RUNNING: "Job is actively running",
            self.CANCELLED: "Job has been cancelled",
            self.DONE: "Job has successfully run",
            self.ERROR: "Job incurred an error",
        }[self]

    @classmethod
    def from_string(cls, status_str: str) -> JobStatus:
        """Create a JobStatus from a string.

        Args:
            status_str (str): The string representation of the status.

        Returns:
            JobStatus: The corresponding JobStatus enum.

        Raises:
            ValueError: If the string doesn't match any JobStatus.
        """
        try:
            return cls(status_str.lower())
        except ValueError:
            msg = f"'{status_str}' is not a valid JobStatus"
            raise ValueError(msg) from None

    @property
    def is_final(self) -> bool:
        """Check if the status is a final state."""
        return self in JOB_FINAL_STATES

    @classmethod
    def non_final_states(cls) -> list[JobStatus]:
        """Return a list of all non-final states."""
        return [status for status in cls if status not in JOB_FINAL_STATES]


JOB_FINAL_STATES: set[JobStatus] = {JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR}


class JobStatusError(Exception):
    """Custom exception for JobStatus-related errors."""

    def __init__(self, message: str, status: JobStatus) -> None:
        self.status = status
        super().__init__(f"JobStatus Error: {message} (Status: {status})")
