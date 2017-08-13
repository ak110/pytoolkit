
import pytoolkit as tk


def test_read_write(tmpdir):
    path = str(tmpdir.join('read_write_test.txt'))

    tk.io.write_all_text(path, 'あああ\nいいい')
    assert tk.io.read_all_text(path) == 'あああ\nいいい'
    assert tk.io.read_all_lines(path) == ['あああ', 'いいい']

    tk.io.write_all_lines(path, ['ううう', 'えええ'])
    assert tk.io.read_all_text(path) == 'ううう\nえええ\n'
    assert tk.io.read_all_lines(path) == ['ううう', 'えええ']
