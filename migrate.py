#!/usr/bin/env python3
"""Migrate existing namehistory.db to include new client profile tables.

Safe to run multiple times — only creates missing tables via SQLAlchemy metadata.
Existing rows in profiles, history, skin_history, users, and requests are preserved.
"""

from main import Base, engine, init_db, log


def migrate() -> None:
    init_db()
    Base.metadata.create_all(bind=engine)
    log.info("Migration complete — client profile tables ensured.")


if __name__ == "__main__":
    migrate()
