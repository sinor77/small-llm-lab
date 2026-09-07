"""
Runs all validation steps and writes a report to validation_report.txt
"""
import subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "validation_report.txt")
lines = []

def run(cmd, cwd=ROOT):
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True,
        timeout=300
    )
    return result.stdout + result.stderr

def section(title):
    lines.append("\n" + "="*60)
    lines.append(title)
    lines.append("="*60)

# 1. Install
section("INSTALL")
out = run([sys.executable, "-m", "pip", "install",
           "tokenizers>=0.15.0", "pytest>=7.4.0",
           "numpy>=1.24.0", "tqdm>=4.66.0", "pyyaml>=6.0"])
# last line of pip output
for l in out.strip().split("\n"):
    if "Successfully" in l or "already" in l or "ERROR" in l:
        lines.append(l)

# 2. Full test suite
section("PYTEST FULL SUITE")
out = run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])
lines.append(out)

# 3. Overfit test only
section("OVERFIT TEST ONLY")
out = run([sys.executable, "-m", "pytest",
           "tests/test_training.py::test_tiny_model_overfits_tiny_dataset",
           "-v", "-s"])
lines.append(out)

with open(REPORT, "w") as f:
    f.write("\n".join(lines))

print("Done. See validation_report.txt")
