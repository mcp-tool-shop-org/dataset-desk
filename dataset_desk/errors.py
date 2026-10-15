"""The desk's structured error: a stable code, a message, a hint, and an exit code.

`DeskError` subclasses `SystemExit` so a refusal stops the program even when it is raised outside the CLI, and
`str(err)` is the message. The CLI entry point (`desk.run`) prints `error <CODE>: <message>` and the hint, never a
stack trace unless `--debug` is passed.

Exit codes: 0 ok, 1 user error (bad input, a refusal, a missing cache), 2 runtime error (an unexpected failure).
"""

from __future__ import annotations

EXIT_OK, EXIT_USER, EXIT_RUNTIME = 0, 1, 2

# The registry. Codes are stable once released.
CODES = {
    "INPUT_BAD_ARGUMENT": "check the command's --help",
    "INPUT_UNKNOWN_BACKEND": "use one of: eg2, siglip2, none, fake",
    "INPUT_UNKNOWN_DRAFTER": "use one of: plain-sight, fake, none",
    "PERM_OUT_INSIDE_PROJECT": "point --out outside the project folder; the desk never writes into a project",
    "PERM_GPU_NEEDS_GRANT": "pass --grant <id> from whoever schedules the GPU, or run on --device cpu",
    "PERM_LOCKED_POOL": "pass --images-only (or --backend none), or --allow-multimodal-pool once the benchmark's "
                        "counted run is finished",
    "DEP_TOKENIZER_MISSING": "install the models extra and cache the tokenizer once, or run without --count-tokens",
    "STATE_NO_INDEX": "run `dataset-desk index` with the same --project, --backend and --out first",
    "RUNTIME_UNEXPECTED": "rerun with --debug for the stack trace, and report it if it persists",
}


class DeskError(SystemExit):
    """A user-facing error with the shape code, message, hint, cause?, retryable?."""

    def __init__(self, code: str, message: str, hint: str | None = None, *, cause: BaseException | None = None,
                 retryable: bool = False, exit_code: int = EXIT_USER) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint if hint is not None else CODES.get(code, "")
        self.cause = cause
        self.retryable = retryable
        self.exit_code = exit_code

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict:
        out = {"code": self.code, "message": self.message, "hint": self.hint, "retryable": self.retryable}
        if self.cause is not None:
            out["cause"] = f"{type(self.cause).__name__}: {self.cause}"
        return out

    def render(self) -> str:
        lines = [f"error {self.code}: {self.message}"]
        if self.hint:
            lines.append(f"hint: {self.hint}")
        if self.cause is not None:
            lines.append(f"cause: {type(self.cause).__name__}: {self.cause}")
        return "\n".join(lines)
