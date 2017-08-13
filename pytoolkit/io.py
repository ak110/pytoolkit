"""ファイル・フォルダ関連"""
import os


def delete_file(file_path):
    """ファイル削除"""
    if os.path.isfile(file_path):
        os.remove(file_path)


def read_all_lines(file_path, mode='r', encoding='utf-8'):
    """ファイルの全行を読み込み。戻り値は改行無しの配列。.NETのSystem.IO.File.ReadAllLine()風。"""
    with open(file_path, mode, encoding=encoding) as f:
        return [l.rstrip('\n') for l in f.readlines()]


def write_all_lines(file_path, lines, mode='w', encoding='utf-8'):
    """ファイルの全行を書き込み。linesは改行無しの文字列の配列。.NETのSystem.IO.File.WriteAllLine()風。"""
    with open(file_path, mode, encoding=encoding) as f:
        f.writelines([l + '\n' for l in lines])


def read_all_text(file_path, mode='r', encoding='utf-8'):
    """ファイルの全行を読み込み。戻り値は改行無しの配列。.NETのSystem.IO.File.ReadAllText()風。"""
    with open(file_path, mode, encoding=encoding) as f:
        return f.read()


def write_all_text(file_path, text, mode='w', encoding='utf-8'):
    """ファイルの全行を書き込み。linesは改行無しの文字列の配列。.NETのSystem.IO.File.WriteAllText()風。"""
    with open(file_path, mode, encoding=encoding) as f:
        f.write(text)
