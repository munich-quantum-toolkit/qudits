from __future__ import annotations

import enum


class JobStatus(enum.Enum):
    """Enumeration for job status."""

    INITIALIZING = "Initializing: Job is being initialized"
    QUEUED = "Queued: Job is waiting in the queue"
    VALIDATING = "Validating: Job is being validated"
    RUNNING = "Running: Job is actively running"
    CANCELLED = "Cancelled: Job has been cancelled"
    DONE = "Done: Job has successfully run"
    ERROR = "Error: Job incurred an error"


JOB_FINAL_STATES = (JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR)
