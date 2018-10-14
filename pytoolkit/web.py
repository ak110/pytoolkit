"""Web関連(主にFlask)。"""
import base64
import datetime
import functools
import pathlib
import secrets
import urllib.parse

from . import log


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
            token = flask.session.get(session_key, None)
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


def get_safe_url(target, host_url, default_url):
    """ログイン時のリダイレクトとして安全なURLを返す。"""
    if target is None or target == '':
        return default_url
    ref_url = urllib.parse.urlparse(host_url)
    test_url = urllib.parse.urlparse(urllib.parse.urljoin(host_url, target))
    if test_url.scheme not in ('http', 'https') or ref_url.netloc != test_url.netloc:
        logger = log.get(__name__)
        logger.warning(f'Invalid next url: {target}')
        return default_url
    return test_url.path


class Paginator:
    """ページング用簡易ヘルパ

    flask-paginateとかもあるがflask依存も面倒なので自作してしまった。

    # 引数

    - page: 1オリジンのページ番号
    - items_per_page: 1ページあたりのアイテム数
    - query: SQLAlchemyのクエリ (itemsといずれか必須)
    - items: 現在のページのデータ (queryといずれか必須)
    - total_items: 全データ件数 (items指定時必須)
    - prev_size: 現在ページ以前の最大表示ページ数
    - next_size: 現在ページ以降の最大表示ページ数

    # 例

    ```html
    {% macro render_pagination(paginator) %}
    <nav>
        <ul class="pagination justify-content-center">
            <li class="page-item{% if not paginator.has_prev %} disabled{% endif %}">
                <a class="page-link" href="{{ url_for('user.index', page=paginator.page - 1) }}">&lt;</a>
            </li>
            {% for page in range(paginator.start_page, paginator.end_page + 1) %}
            <li class="page-item{% if paginator.page == page %} active{% endif %}">
                <a class="page-link" href="{{ url_for('user.index', page=page) }}">{{ page }}</a>
            </li>
            {% endfor %}
            <li class="page-item{% if not paginator.has_next %} disabled{% endif %}">
                <a class="page-link" href="{{ url_for('user.index', page=paginator.page + 1) }}">&gt;</a>
            </li>
        </ul>
    </nav>
    {% endmacro %}
    ```

    """

    def __init__(self, page, items_per_page, query=None, items=None, total_items=None, prev_size=3, next_size=3):
        assert page >= 1
        assert items_per_page >= 1
        self.page = page
        self.items_per_page = items_per_page
        if query is not None:
            assert items is None
            self.total_items = query.count()
            self.items = query.slice(items_per_page * (page - 1), items_per_page * page)
        else:
            assert items is not None
            assert total_items >= 0
            self.items = items
            self.total_items = total_items
        self.prev_size = prev_size
        self.next_size = next_size

    @property
    def pages(self):
        """ページ数。"""
        if self.total_items <= 0:
            return 1
        return (self.total_items + self.items_per_page - 1) // self.items_per_page

    @property
    def start_page(self):
        """開始ページ。"""
        return max(1, self.page - self.prev_size)

    @property
    def end_page(self):
        """終了ページ。"""
        return min(self.pages, self.page + self.next_size)

    @property
    def has_prev(self):
        """前ページがあるか否か。"""
        return self.page > 1

    @property
    def has_next(self):
        """次ページがあるか否か。"""
        return self.page < self.pages
