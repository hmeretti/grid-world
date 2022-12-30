from utils.operations import (
    add_tuples,
    float_dict_comparison,
    order_dict,
    order_callable,
)

d1 = {"a": 1.02, "b": 2.31, "c": 2}
ordered_d1 = [("b", 2.31), ("c", 2), ("a", 1.02)]
d2 = {"a": 1.02, "b": 2.31, "c": 2 + 1e-20}
d3 = {"a": 1.02, "b": 2.31, "c": 2.3}
d4 = {"a": 1.02, "b": 2.31}
d5 = {"a": 1.02, "b": 2.31, "f": 2 + 1e-20}


class TestUtils:
    @staticmethod
    def test_add_tuples():
        result = add_tuples((1, 2), (2, 3))
        assert result == (3, 5)
        assert isinstance(result, tuple)
        assert all([isinstance(x, int) for x in result])

    @staticmethod
    def test_float_dict_comparison_equality():
        assert float_dict_comparison(d1, d1)
        assert float_dict_comparison(d1, d2)
        assert float_dict_comparison(d2, d1)

    @staticmethod
    def test_float_dict_comparison_different_values():
        assert not float_dict_comparison(d1, d3)
        assert not float_dict_comparison(d3, d1)

    @staticmethod
    def test_float_dict_comparison_different_keys():
        assert not float_dict_comparison(d1, d4)
        assert not float_dict_comparison(d4, d1)
        assert not float_dict_comparison(d1, d5)
        assert not float_dict_comparison(d5, d1)

    @staticmethod
    def test_order_dict():
        assert order_dict(d1) == ordered_d1

    @staticmethod
    def test_order_callable():
        assert order_callable(lambda x: d1.get(x, 0), list(d1.keys())) == ordered_d1
        assert order_callable(lambda x: x**2, [2, 1, 3]) == [(3, 9), (2, 4), (1, 1)]
        assert order_callable(lambda x: x**2, [2, 1, 3], reverse=False) == [
            (1, 1),
            (2, 4),
            (3, 9),
        ]
