import pathlib

import pytoolkit as tk


def test_read_write(tmpdir):
    path = pathlib.Path(tmpdir.join('read_write_test.txt'))

    path.write_text('あああ\nいいい', encoding='utf-8')
    assert path.read_text(encoding='utf-8') == 'あああ\nいいい'
    assert tk.io.read_all_lines(path) == ['あああ', 'いいい']

    tk.io.write_all_lines(path, ['ううう', 'えええ'])
    assert path.read_text(encoding='utf-8') == 'ううう\nえええ\n'
    assert tk.io.read_all_lines(path) == ['ううう', 'えええ']
