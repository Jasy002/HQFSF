"""
Unit Test for Helper Module.
"""

import utils.helper as helper


def test_helper_module():

    print("\n========== Helper Module ==========\n")

    print("Helper module imported successfully.")

    assert helper is not None

    print("\n✓ Helper Test Passed")


if __name__ == "__main__":
    test_helper_module()