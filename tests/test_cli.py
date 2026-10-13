"""CLI 冒烟：densities / selftest / --version。"""
import pytest

from rectforge import __version__
from rectforge.cli import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as ei:
        main(["--version"])
    assert ei.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_densities_command():
    assert main(["densities"]) == 0


def test_selftest_command():
    assert main(["selftest"]) == 0
