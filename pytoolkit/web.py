"""Web関連(主にFlask)。"""
import base64


def data_url(data: bytes, mime_type: str) -> str:
    """小さい画像などのバイナリデータをURLに埋め込んだものを作って返す。

    # 引数
    - data: 埋め込むデータ
    - mime_type: 例：'image/png'

    """
    b64 = base64.b64encode(data).decode('ascii')
    return f'data:{mime_type};base64,{b64}'
