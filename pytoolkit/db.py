"""DB(主にSQLAlchemy)関連"""
import logging
import time

from . import utils


def register_ping():
    """コネクションプールの切断対策。"""
    import sqlalchemy

    @sqlalchemy.event.listens_for(sqlalchemy.pool.Pool, 'checkout')
    def _ping_connection(dbapi_connection, connection_record, connection_proxy):
        utils.noqa(connection_record, connection_proxy)
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute('SELECT 1')
        except BaseException:
            raise sqlalchemy.exc.DisconnectionError()
        finally:
            cursor.close()


def wait_for_connection(url, timeout=10):
    """DBに接続可能になるまで待機する。"""
    import sqlalchemy
    logger = logging.getLogger(__name__)
    failed = False
    start_time = time.time()
    while True:
        try:
            engine = sqlalchemy.create_engine(url)
            try:
                result = engine.execute('SELECT 1')
                try:
                    # 接続成功
                    if failed:
                        logger.info('DB接続成功')
                    break
                finally:
                    result.close()
            finally:
                engine.dispose()
        except BaseException:
            # 接続失敗
            if not failed:
                failed = True
                logger.info(f'DB接続待機中 . . . (URL: {url})')
            if time.time() - start_time >= timeout:
                raise
            time.sleep(1)


def safe_close(session):
    """例外を無視するclose。"""
    try:
        session.close()
    except BaseException:
        pass


def long_text_type():
    """LONGTEXTな型を返す。"""
    import sqlalchemy.dialects.mysql
    import sqlalchemy.dialects.sqlite
    import sqlalchemy.sql.sqltypes
    t = sqlalchemy.sql.sqltypes.Text()
    t = t.with_variant(sqlalchemy.dialects.mysql.LONGTEXT(), 'mysql')
    t = t.with_variant(sqlalchemy.dialects.sqlite.TEXT(), 'sqlite')
    return t


def big_int_type():
    """BIGINTな型を返す。"""
    import sqlalchemy.dialects.mysql
    import sqlalchemy.dialects.sqlite
    import sqlalchemy.sql.sqltypes
    t = sqlalchemy.sql.sqltypes.Integer()
    t = t.with_variant(sqlalchemy.dialects.mysql.BIGINT(), 'mysql')
    t = t.with_variant(sqlalchemy.dialects.sqlite.INTEGER(), 'sqlite')
    return t
