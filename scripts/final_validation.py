"""
Final validation: run full pytest suite + pipeline validation, write combined report.
"""
import subprocess, sys, os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT  = os.path.join(ROOT, "final_validation_report.txt")

def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return r.stdout + r.stderr

sections = []

def section(t, content):
    sections.append(f"\n{'='*65}\n{t}\n{'='*65}\n{content}")

pytest_out  = run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])
section("PYTEST FULL TEST SUITE", pytest_out)

pipeline_out = run([sys.executable, "scripts/validate_pipeline.py"])
section("PIPELINE VALIDATION", pipeline_out)

full = "\n".join(sections)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(full)

print(f"Report written to: {OUT}")

# Print key summary lines
for line in full.split("\n"):
    if any(k in line for k in [
        "passed", "failed", "error", "[PASS]", "[FAIL]", "[INFO]",
        "SUMMARY", "===", "Overfit", "step/sec", "tokens/sec", "minutes"
    ]):
        print(line)
