import pytest

from adpa import PackageASTAnalyzer


def test_analyzer_creation():
    analyzer = PackageASTAnalyzer("test_module", "test_package")
    assert analyzer.module_name == "test_module"
    assert analyzer.package_name == "test_package"
