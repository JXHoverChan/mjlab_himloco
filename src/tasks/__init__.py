"""Task registrations included in this Go2 HIM project."""

# Keep the published repository focused on the runnable Go2 HIM task. The
# generic mjlab task importer would also discover local task experiments that
# are intentionally not part of this repository.
from src.tasks.velocity.config import go2 as _go2  # noqa: F401
