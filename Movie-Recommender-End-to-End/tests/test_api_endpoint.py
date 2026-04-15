import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.serve import api as api_mod


@pytest.fixture(scope="module")
def client():
    with TestClient(api_mod.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["items"] > 0


def test_recommend_known_user(client):
    uid = int(api_mod.store.user_ids[0])
    r = client.get(f"/recommend/{uid}", params={"k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["userId"] == uid
    assert len(body["items"]) == 5


def test_recommend_unknown_user_404(client):
    r = client.get("/recommend/999999999", params={"k": 5})
    assert r.status_code == 404


def test_similar_known_movie(client):
    mid = int(api_mod.store.item_ids[0])
    r = client.get(f"/similar/{mid}", params={"k": 5})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 5
    assert all(it["movieId"] != mid for it in items)


def test_cold_start(client):
    liked = [int(x) for x in api_mod.store.item_ids[:3]]
    r = client.post("/cold_start", json={"liked_movie_ids": liked, "k": 5})
    assert r.status_code == 200
    assert len(r.json()["items"]) == 5
