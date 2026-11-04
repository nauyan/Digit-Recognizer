from rolling_big.registry import list_functions


def test_list_functions_not_empty():
    assert list_functions()
