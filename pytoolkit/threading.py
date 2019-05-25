"""スレッド関連。"""
import concurrent.futures

_pool = concurrent.futures.ThreadPoolExecutor()


def get_pool():
    """色々使いまわす用スレッドプール。"""
    return _pool
