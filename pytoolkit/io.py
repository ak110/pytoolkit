"""ファイル・フォルダ関連"""
import pathlib
import time


def open_devnull(mode='r', buffering=-1, encoding=None, errors=None, newline=None):
    """`os.devnull`を開いて返す。"""
    import os
    return pathlib.Path(os.devnull).open(mode, buffering, encoding, errors, newline)


def delete_file(file_path):
    """ファイル削除"""
    file_path = pathlib.Path(file_path)
    if file_path.is_file():
        file_path.unlink()


def read_all_lines(file_path, mode='r', encoding='utf-8'):
    """ファイルの全行を読み込み。戻り値は改行無しの配列。.NETのSystem.IO.File.ReadAllLine()風。"""
    with pathlib.Path(file_path).open(mode, encoding=encoding) as f:
        return [l.rstrip('\n') for l in f.readlines()]


def write_all_lines(file_path, lines, mode='w', encoding='utf-8'):
    """ファイルの全行を書き込み。linesは改行無しの文字列の配列。.NETのSystem.IO.File.WriteAllLine()風。"""
    with pathlib.Path(file_path).open(mode, encoding=encoding) as f:
        f.writelines([l + '\n' for l in lines])


def do_retry(func, count=10, sleep_seconds=1.0):
    """リトライ処理。"""
    retry = 0
    while True:
        try:
            return func()
        except BaseException:
            if retry >= count:
                raise
            retry += 1
            time.sleep(sleep_seconds)


def get_all_files(dir_path):
    """ファイルの列挙。"""
    import os
    for root, _, files in os.walk(str(dir_path)):
        parent = pathlib.Path(root)
        for file in files:
            yield parent / file


def get_all_entries(dir_path):
    """ファイル・ディレクトリの列挙。"""
    import os
    for root, _, files in os.walk(str(dir_path)):
        parent = pathlib.Path(root)
        yield parent
        for file in files:
            yield parent / file


def get_size(path):
    """ファイル・ディレクトリのサイズを返す。"""
    path = pathlib.Path(path)
    if path.is_dir():
        return sum([p.stat().st_size for p in path.glob('**/*') if not p.is_dir()])
    else:
        return path.stat().st_size
