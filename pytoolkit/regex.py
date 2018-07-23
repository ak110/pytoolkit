"""正規表現関連。"""
import re

IGNORECASE = re.IGNORECASE
MULTILINE = re.MULTILINE


def is_match(pattern, string, flags=0):
    """一致判定。"""
    return re.match(pattern, string, flags) is not None


def search(pattern, string, flags=0):
    """検索。

    matchオブジェクトを返す。bool扱い出来て、
    - m.group(0) : 全体
    - m.group(1) : 最初のサブグループ
    のようにサブグループも取れる。
    """
    return re.search(pattern, string, flags)


def search_all(pattern, string, flags=0):
    """検索。

    matchオブジェクトのiteratorを返す。
    """
    return re.finditer(pattern, string, flags)


def replace(pattern, repl, string, flags=0):
    """置換。"""
    return re.sub(pattern, repl, string, flags)
