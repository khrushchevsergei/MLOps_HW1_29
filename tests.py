import os
import pytest
from pytest_postgresql import factories
from sqlalchemy.exc import ProgrammingError
from clothes_shop import application_setup
import Basic_Trainer

# Tests for database


def get_flats(db_url):
    """Return rows from the dataset table."""
    sql = "SELECT * FROM flat.pricess"
    results = pd.read_sql_query(sql=sql, con=db_url)
    if len(results) > 1:
        # Do logic here.
        pass
    return results


@pytest.fixture()
def database(postgresql):
    """Set up the mock DB with the SQL flat file."""
    with open("test.sql") as f:
        setup_sql = f.read()
    with postgresql.cursor() as cursor:
        cursor.execute(setup_sql)
        postgresql.commit()
    yield postgresql
def test_example_postgres(database):
    drivers = get_drivers(db_url=database)
    assert len(flats) == 2
    assert set(flats["id"]) == {"1", "2"}


postgresql_in_docker = factories.postgresql_noproc()
postgresql = factories.postgresql("postgresql_in_docker", dbname="sample_db")


def test_invalid_user():
    """Check the database connection does not allow an invalid username"""
    os.environ["PG_USERNAME"] = "invalid_username"
    os.environ["DATABASE_NAME"] = "mlops"
    table_name = "train_flat_prices"

    app, db = application_setup.application_setup()

    assert app.config["SQLALCHEMY_DATABASE_URI"] == str(db.session.bind.url)
    with pytest.raises(ProgrammingError):
        assert db.session.execute(f"SELECT * from {table_name}")

def test_postgres_docker(postgresql):
    """Run test."""
    cur = postgresql.cursor()
    cur.execute("CREATE TABLE test (id serial PRIMARY KEY, num integer, data varchar);")
    postgresql.commit()
    cur.close()



#Tests for models 


def tcorrect_predict_form_GB():
    Dataset = pd.read_csv('Dataset/train_flat_prices.csv')
    model = Basic_Trainer(1, 'GradientBoostingClassifier', Dataset)
    model.fit({'random_state': 10})
    assert len(model.predict())!=0

def correct_predict_form_LR():
    Dataset = pd.read_csv('Dataset/train_flat_prices.csv')
    model = Basic_Trainer(2, 'LogisticRegression', Dataset)
    model.fit({'random_state': 10})
    assert len(model.predict())!=0


