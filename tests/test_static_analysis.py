import subprocess
import sys

def test_static_analysis() -> None:
    """Run mypy to verify type correctness across src, tests, and scripts."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "src", "tests", "scripts", "--ignore-missing-imports"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, (
        f"mypy failed with exit code {result.returncode}:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
