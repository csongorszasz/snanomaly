import subprocess
import sys


# ruff: noqa: T201, S603, S607
def main() -> None:
    """Run all linting and type checking tools."""
    print("Running ruff...")
    ruff = subprocess.run(["ruff", "check", "src", "tests"], check=False)

    print("\nRunning pylint...")
    pylint = subprocess.run(["pylint", "src"], check=False)

    print("\nRunning mypy...")
    mypy = subprocess.run(["mypy", "src"], check=False)

    # Return non-zero if any tool failed
    sys.exit(ruff.returncode or pylint.returncode or mypy.returncode)


if __name__ == "__main__":
    main()
