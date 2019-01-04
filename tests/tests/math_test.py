
import pytoolkit as tk


def test_format_values():
    assert tk.math.format_values([0.1, +1]) == ['0.100', '1.000']
    assert tk.math.format_values([0.1, -1]) == [' 0.100', '-1.000']
    assert tk.math.format_values([0.1, 1, 10]) == ['1.00e-01', '1.00e+00', '1.00e+01']
    assert tk.math.format_values([0.01, 0.02]) == ['1.00e-02', '2.00e-02']
    assert tk.math.format_values([-9, 99, 999]) == ['  -9', '  99', ' 999']
