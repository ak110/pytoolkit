"""DB(主にSQLAlchemy)関連"""

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


def safe_close(session):
    """例外を無視するclose。"""
    try:
        session.close()
    except BaseException:
        pass
