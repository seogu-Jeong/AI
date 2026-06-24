from pathlib import Path


def test_compose_creates_test_database_before_migrations():
    compose = Path("docker-compose.yml").read_text()

    assert "db-init:" in compose
    assert "POSTGRES_TEST_DB" in compose
    assert "stocksense_test" in compose
    assert "createdb" in compose
    assert "TEST_DATABASE_URL:" in compose
    assert "${POSTGRES_TEST_DB:-stocksense_test}" in compose
    assert "db-init:" in compose.split("migrate:", 1)[1]
    assert "condition: service_completed_successfully" in compose.split("migrate:", 1)[1]
