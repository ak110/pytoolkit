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
        with dbapi_connection.cursor() as cursor:
            try:
                cursor.execute('SELECT 1')
            except BaseException:
                raise sqlalchemy.exc.DisconnectionError()


def wait_for_connection(url):
    """DBに接続可能になるまで待機する。"""
    import sqlalchemy
    logger = logging.getLogger(__name__)
    failed = False
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
            time.sleep(1)


def safe_close(session):
    """例外を無視するclose。"""
    try:
        session.close()
    except BaseException:
        pass
