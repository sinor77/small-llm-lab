"""Wrapper: run sanity_experiment.py and capture all output to files."""
import subprocess, sys, os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

r = subprocess.run(
    [sys.executable, "scripts/sanity_experiment.py"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    timeout=600,
    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
)
out = r.stdout + r.stderr

with open(os.path.join(ROOT, "sanity_out.txt"), "w", encoding="utf-8") as f:
    f.write(out)

print(out[-4000:] if len(out) > 4000 else out)
print(f"\nReturn code: {r.returncode}")
