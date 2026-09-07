"""Run tests and write results to a file."""
import subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "test_results.txt")

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
    cwd=ROOT, capture_output=True, text=True, timeout=300
)
output = result.stdout + result.stderr
with open(OUT, "w") as f:
    f.write(output)
print(output[-3000:])  # last 3000 chars to stdout
