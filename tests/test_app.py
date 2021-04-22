from app import get_hash, verify_hash, DB
import os
import re
import pytest


def test_get_hash():
    assert get_hash("Hello") != "Hello"
    assert get_hash("strive") != "strive"


def text_bcrypt():

    bcrypt_hash_pattern = r"^\$2[ayb]\$.{56}$"
    pat = re.compile(bcrypt_hash_pattern)

    new_hash = get_hash("Hello")

    assert pat.search(new_hash)


@pytest.mark.parametrize("testing_string", [("hello"), ("strive"), ("asdasd")])
def test_hash_computation(testing_string):
    # testing_string = "hello"
    new_hash = get_hash(testing_string)

    assert verify_hash(testing_string, new_hash)


@pytest.mark.parametrize(
    "email,name, password", [("hello@hello.com", "steve", "asd"), ("strive@strive.com", "elon", "asdasd")]
)
def test_user_creation(email, name, password):

    # We are not checking if an email already exists so
    # we can have the same email but *the password hashes
    # will be different*. For that reason, I'm deleting the whole
    # users table on every new test

    # BE CAREFUL: If I were running this test on a "production" database, it would break
    # everything. That's why I'm creating a new db called "testing.db" for every test.

    db = DB(dbname="testing.db")

    # remove eerything from the users table
    db.conn.execute("DELETE FROM users")
    db.conn.execute("COMMIT")

    # create a new user (password will be hashed inside the function)
    user_id = db.create_user(name, email, password)

    # here I'm using the internal DB connection in the class to send a custom
    # SQL query
    # I'm doing this to get the email and the hashed password of the user we just
    # created. The email should be the same one we have used in the
    # db.create_user() function
    result = db.conn.execute(
        "SELECT * from users where user_id = ?", (user_id,)
    ).fetchone()

    # result is a tuple with 3 elements:
    # (user_id, email, hashed_password)

    user_id = result[0]
    new_user_name = result[1]
    new_user_email = result[2]
    hashed_password = result[3]

    # make sure the emails match
    assert new_user_email == email

    assert db.validate_password(email, password)

    # examples of what to test
    # 1. verify user is created
    # 2. verify you can validate the hash
    # 3. whatever you come up with
