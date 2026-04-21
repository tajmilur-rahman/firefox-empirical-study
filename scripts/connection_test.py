from src import db_connection

# Test inserting a single sample bug
sample_bug = {"id": 12345, "title": "test,example"}
db_connection.save(sample_bug)
