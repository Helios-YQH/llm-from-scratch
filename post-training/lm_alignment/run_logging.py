"""Where a run's artifacts live, and how they get recorded.

    <run_dir>/
        config.json      hyperparameters, environment, git revision, command line
        console.log      everything the run printed, including tracebacks
        metrics.jsonl    one JSON object per step -- the durable source of truth
        summary.json     best/final values per metric, written on a clean close
        error.txt        traceback, written if the run died
        tensorboard/     event files, for `tensorboard --logdir`
        rollouts/        sampled generations, one file per dump
        checkpoints/     policy snapshots, with optimizer state for `--resume`
        final/           the trained policy
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


def _iso(epoch_seconds: float | None) -> str | None:
    if epoch_seconds is None:
        return None
    return datetime.fromtimestamp(epoch_seconds).astimezone().isoformat(timespec="seconds")


def _git_state() -> dict[str, Any]:
    """Commit hash and whether the tree was dirty when the run started."""
    repo = Path(__file__).resolve().parent.parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=False
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    if not commit:
        return {}
    return {"commit": commit, "dirty": bool(git("status", "--porcelain"))}


def _environment() -> dict[str, Any]:
    import torch

    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        environment["gpus"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    for module_name in ("transformers", "vllm"):
        try:
            module = __import__(module_name)
            environment[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            environment[module_name] = None
    return environment


def build_config(params: dict[str, Any]) -> dict[str, Any]:
    """Collect the run's config. Call with `locals()` so no flag goes unrecorded."""
    return {
        "command": " ".join(sys.argv),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hyperparameters": {k: v for k, v in params.items() if not k.startswith("_")},
        "git": _git_state(),
        "environment": _environment(),
    }


class _Tee:
    """Write to the real stream and to the run's log file at once."""

    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._log_file.write(data)
        self._log_file.flush()
        return self._stream.write(data)

    def flush(self) -> None:
        self._log_file.flush()
        self._stream.flush()

    def isatty(self) -> bool:
        # Progress bars then print one line per update instead of repainting,
        # which is what you want in a file that is read later.
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class RunLogger:
    """Fans one step's metrics out to every enabled sink."""

    def __init__(
        self,
        run_dir: str | Path,
        config: dict[str, Any],
        tensorboard: bool = True,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        capture_console: bool = True,
        overwrite: bool = False,
        resume: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self._metrics_path = self.run_dir / "metrics.jsonl"
        if self._metrics_path.exists() and not (overwrite or resume):
            raise FileExistsError(
                f"{self._metrics_path} already exists; a rerun would silently append to the "
                f"old run's metrics. Pass --resume to continue it, --overwrite-run to discard it, "
                f"or delete {self.run_dir}."
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("rollouts", "checkpoints"):
            (self.run_dir / subdir).mkdir(exist_ok=True)

        if not resume:
            # A fresh run must not inherit the previous attempt's artifacts. An
            # `error.txt` left by a crash reads as though *this* run died, and a
            # stale event file puts the discarded run's curves back on the
            # dashboard. `--resume` is the only mode that should keep the history.
            (self.run_dir / "error.txt").unlink(missing_ok=True)
            for stale in (self.run_dir / "tensorboard").glob("events.out.tfevents.*"):
                stale.unlink()

        # Keep the original: it carries the commit that produced the early steps.
        config_path = self.run_dir / "config.json"
        if resume and config_path.exists():
            config_path = self.run_dir / f"config.resume.{datetime.now():%Y%m%d_%H%M%S}.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

        # "a" keeps a resumed run's earlier steps.
        self._metrics_file = self._metrics_path.open(
            "a" if resume else "w", encoding="utf-8"
        )

        self._console_file = None
        self._saved_streams = None
        if capture_console:
            self._console_file = (self.run_dir / "console.log").open("a", encoding="utf-8")
            self._saved_streams = (sys.stdout, sys.stderr)
            sys.stdout = _Tee(self._saved_streams[0], self._console_file)
            sys.stderr = _Tee(self._saved_streams[1], self._console_file)
            print(f"[run] artifacts in {self.run_dir}")

        # Catches failures anywhere in the script, not just inside the loop.
        self._previous_excepthook = sys.excepthook
        sys.excepthook = self._excepthook

        # SIGKILL (what the OOM killer sends) cannot be caught; this covers the
        # polite kills.
        self._previous_sigterm = None
        try:
            import signal

            self._previous_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._excepthook_sigterm)
        except (ImportError, ValueError, OSError):
            self._previous_sigterm = None  # not the main thread, or unsupported

        self._writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            except ImportError:
                print("tensorboard not installed; skipping the dashboard sink.")

        self._wandb_run = None
        if wandb_project:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=self.run_dir.name,
                    config=config,
                    resume="allow" if resume else None,
                    # The machine is shared: wandb's system monitor would chart every
                    # GPU on the box, five of which belong to other people, and its
                    # once-a-second sampling also advances the run's step counter --
                    # which is what squeezed the training curves into the first few
                    # pixels of each plot.
                    settings=wandb.Settings(x_disable_stats=True, x_disable_meta=True),
                )
                # Belt and braces: name the rollout step as the charts' x-axis.
                self._wandb_run.define_metric("*", step_metric="rollout_step")
            except Exception as error:
                # A logging backend must never take down an hours-long run.
                print(f"wandb init failed ({error}); continuing without it.")

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        """Record one step. Non-numeric values are kept in the JSONL only."""
        # elapsed_s resets across a resume, so keep an absolute clock too.
        self._metrics_file.write(
            json.dumps({"step": step, "wall_time": time.time(), **metrics}) + "\n"
        )
        self._metrics_file.flush()

        scalars = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if self._writer is not None:
            for name, value in scalars.items():
                self._writer.add_scalar(name, value, step)
        if self._wandb_run is not None:
            self._wandb_run.log({**scalars, "rollout_step": step})

    @property
    def console_fileno(self) -> int | None:
        """console.log's fd. Subprocesses inherit fd 1/2, not sys.stdout."""
        return None if self._console_file is None else self._console_file.fileno()

    def record_error(self, error: BaseException) -> None:
        """Persist a traceback next to the run it killed."""
        with (self.run_dir / "error.txt").open("a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.now().astimezone().isoformat(timespec='seconds')} ===\n")
            traceback.print_exception(type(error), error, error.__traceback__, file=f)

    def _excepthook(self, exc_type, exc, tb) -> None:
        # Write a summary first: the run is ending early, but the steps it did
        # complete are still worth comparing against other runs.
        self.record_error(exc)
        try:
            self._write_summary()
        except Exception:
            pass
        self._previous_excepthook(exc_type, exc, tb)

    def _excepthook_sigterm(self, signum, frame) -> None:
        # Deliberately minimal: the handler can interrupt a write, so it records
        # the signal and gets out rather than trying to close everything.
        self.record_error(SystemExit(f"terminated by signal {signum}"))
        raise SystemExit(128 + signum)

    def rollout_path(self, label: str) -> Path:
        return self.run_dir / "rollouts" / f"{label}.txt"

    def checkpoint_dir(self, label: str) -> Path:
        return self.run_dir / "checkpoints" / label

    def latest_checkpoint(self) -> Path | None:
        """Newest `stepNNNN` holding a training_state.pt; without one it is a partial write."""
        candidates = [
            path
            for path in sorted((self.run_dir / "checkpoints").glob("step[0-9]*"))
            if (path / "training_state.pt").is_file()
        ]
        return candidates[-1] if candidates else None

    def _write_summary(self) -> None:
        """One-line-per-metric overview, so comparing runs needs no JSONL parsing."""
        records = []
        for line in self._metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A kill mid-write leaves a torn final line; skip it rather than
                # losing the whole summary.
                continue
        if not records:
            return
        numeric_keys = {
            k for record in records for k, v in record.items() if isinstance(v, (int, float))
        } - {"step", "wall_time"}
        first_wall, last_wall = records[0].get("wall_time"), records[-1].get("wall_time")
        summary: dict[str, Any] = {
            "n_steps": len(records),
            "last_step": records[-1]["step"],
            "started_wall": _iso(first_wall),
            "ended_wall": _iso(last_wall),
            "wall_time_s": records[-1].get("elapsed_s"),
            "final": {k: records[-1][k] for k in numeric_keys if k in records[-1]},
            "per_metric": {},
        }
        for key in sorted(numeric_keys):
            values = [(r["step"], r[key]) for r in records if key in r]
            if not values:
                continue
            best_step, best_value = max(values, key=lambda sv: sv[1])
            summary["per_metric"][key] = {
                "first": values[0][1],
                "last": values[-1][1],
                "min": min(v for _, v in values),
                "max": max(v for _, v in values),
                "max_step": best_step,
            }
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

    def close(self) -> None:
        if sys.excepthook == self._excepthook:
            sys.excepthook = self._previous_excepthook
        if self._previous_sigterm is not None:
            import signal

            signal.signal(signal.SIGTERM, self._previous_sigterm)
            self._previous_sigterm = None
        if self._console_file is not None and self._saved_streams is not None:
            sys.stdout, sys.stderr = self._saved_streams
            self._console_file.close()
        try:
            self._write_summary()
        except Exception as error:  # the summary is a convenience, not the record
            print(f"could not write summary.json ({error})")
        self._metrics_file.close()
        if self._writer is not None:
            self._writer.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
