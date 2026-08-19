from __future__ import annotations

import json
import os
from typing import Any

import firebase_admin
from firebase_admin import credentials, db


class FirebaseStore:
    """Small Firebase Realtime Database adapter.

    The service-account JSON is read only from the FIREBASE_SERVICE_ACCOUNT
    environment variable. It is never persisted by the application.
    """

    def __init__(self, database_url: str, service_account_json: str | None = None) -> None:
        self.database_url = database_url
        self.service_account_json = service_account_json or os.getenv("FIREBASE_SERVICE_ACCOUNT")
        self._initialize()

    def _initialize(self) -> None:
        if not self.database_url:
            raise ValueError("Firebase database URL is required")
        if not self.service_account_json:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT is required")

        try:
            service_account = json.loads(self.service_account_json)
        except json.JSONDecodeError as exc:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT is not valid JSON") from exc

        if not firebase_admin._apps:
            firebase_admin.initialize_app(
                credentials.Certificate(service_account),
                {"databaseURL": self.database_url},
            )

    def write(self, path: str, value: Any) -> None:
        if not path or path.startswith("/"):
            raise ValueError("path must be a non-empty relative Firebase path")
        db.reference(path).set(value)

    def push(self, path: str, value: Any) -> str:
        if not path or path.startswith("/"):
            raise ValueError("path must be a non-empty relative Firebase path")
        ref = db.reference(path).push(value)
        return ref.key or ""
