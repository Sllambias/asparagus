def test_smoke():
    """Smoke test: Verify that 1+1 is 2."""
    assert 1 + 1 == 2


def test_import_project():
    """Verify that the project code is accessible to tests."""
    try:
        import asparagus  # noqa: F401, I001

        assert True
    except ImportError:
        assert False, "The 'asparagus' package is not accessible from tests"
