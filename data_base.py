
import psycopg2
import os
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE = {
    'drivername': 'postgresql',
    'host': 'mlops_db',
    'port': '5432',
    'username': 'mlops',
    'password': 'ops',
    'database': 'sample_db'
}

engine = create_engine(URL.create(**DATABASE))


def Postgres_database():
    engine = create_engine('postgresql://sergeikhrushchev@localhost:5432/sample_db')
    df = pd.DataFrame(columns=['model_ID', 'Model_Class'])
    df.to_sql('models', con=engine, if_exists='replace')

engine = Postgres_database()
