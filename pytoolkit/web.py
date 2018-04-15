"""Web関連(主にFlask)。"""
import base64
import datetime
import functools
import pathlib
import secrets


def generate_secret_key(cache_path):
    """シークレットキーの作成/取得。

    既にcache_pathに保存済みならそれを返し、でなくば作成する。
    """
    cache_path = pathlib.Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open('a+b') as secret:
        secret.seek(0)
        secret_key = secret.read()
        if not secret_key:
            secret_key = secrets.token_bytes()
            secret.write(secret_key)
            secret.flush()
        return secret_key


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

    def _csrf_nocache(r):
        if session_key in flask.session:
            set_no_cache(r)
        return r

    def _generate_csrf_token():
        if session_key not in flask.session:
            flask.session[session_key] = secrets.token_hex()
        return flask.session[session_key]

    app.before_request(_csrf_protect)
    app.after_request(_csrf_nocache)
    app.jinja_env.globals[func_name] = _generate_csrf_token


def nocache(action):
    """キャッシュさせないようにするデコレーター。"""
    import flask

    @functools.wraps(action)
    def _nocache(*args, **kwargs):
        response = flask.make_response(action(*args, **kwargs))
        set_no_cache(response)
        return response

    return _nocache


def set_no_cache(response):
    """Responseをキャッシュしないように設定する。"""
    response.headers['Last-Modified'] = datetime.datetime.now()
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'


def data_url(data: bytes, mime_type: str) -> str:
    """小さい画像などのバイナリデータをURLに埋め込んだものを作って返す。

    # 引数
    - data: 埋め込むデータ
    - mime_type: 例：'image/png'

    """
    b64 = base64.b64encode(data).decode('ascii')
    return f'data:{mime_type};base64,{b64}'
