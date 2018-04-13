"""Web関連(主にFlask)。"""
import base64
import secrets


def register_csrf_token(app, session_key='_csrf_token', form_key='nonce', func_name='csrf_token'):
    """CSRF対策の処理を登録する。

    # 使用例

    POSTなformに以下を入れる。

    ```html
    <input type="hidden" name="nonce" value="{{ csrf_token() }}" />
    ```

    """
    import flask

    def _csrf_protect():
        if flask.request.method == 'POST':
            token = flask.session.pop(session_key, None)
            if not token or token != flask.request.form.get(form_key):
                flask.abort(403)

    def _generate_csrf_token():
        if session_key not in flask.session:
            flask.session[session_key] = secrets.token_hex()
        return flask.session[session_key]

    app.before_request(_csrf_protect)
    app.jinja_env.globals[func_name] = _generate_csrf_token


def data_url(data: bytes, mime_type: str) -> str:
    """小さい画像などのバイナリデータをURLに埋め込んだものを作って返す。

    # 引数
    - data: 埋め込むデータ
    - mime_type: 例：'image/png'

    """
    b64 = base64.b64encode(data).decode('ascii')
    return f'data:{mime_type};base64,{b64}'
