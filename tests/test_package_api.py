"""End-to-end API behavior for atomic offline-device package uploads.

Covers the in-memory backend (the shared suite fixture): out-of-order
in-package causality, whole-package validation/conflict rejection with no
partial writes, per-item accepted/duplicate statuses, package-level replay
returning the first saved result, and reset interaction.
"""


def _rev(rev, parent, start, end, value=None, withdrawn=False, key=None,
         sensor="S1", unit="C"):
    return {
        "sensor_id": sensor,
        "revision_id": rev,
        "parent_id": parent,
        "idempotency_key": key or f"k-{rev}",
        "start_ns": start,
        "end_ns": end,
        "value": value,
        "unit": unit,
        "withdrawn": withdrawn,
    }


def _package(package_id, revisions, batch="B1"):
    return {
        "batch_id": batch,
        "package_id": package_id,
        "revisions": revisions,
    }


def _post_package(client, body, batch="B1"):
    return client.post(f"/api/v1/batches/{batch}/packages", json=body)


def _post_event(client, rev, parent, start, end, value=None, key=None,
                batch="B1", sensor="S1", unit="C", withdrawn=False):
    return client.post(
        f"/api/v1/batches/{batch}/events",
        json={
            "batch_id": batch,
            "sensor_id": sensor,
            "revision_id": rev,
            "parent_id": parent,
            "idempotency_key": key or f"k-{rev}",
            "start_ns": start,
            "end_ns": end,
            "value": value,
            "unit": unit,
            "withdrawn": withdrawn,
        },
    )


# ---------------------------------------------------------------------------
# Happy path: atomic commit, causal fold of the merged package
# ---------------------------------------------------------------------------


def test_package_commit_folds_out_of_order_parents(client):
    # Child listed before its parent: the merged package folds by causality.
    body = _package("PKG-1", [
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
    ])
    r = _post_package(client, body)
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "accepted"
    assert payload["batch_id"] == "B1"
    assert payload["package_id"] == "PKG-1"
    # Items come back in request order, each newly stored.
    assert [(i["revision_id"], i["status"]) for i in payload["items"]] == [
        ("v2", "accepted"),
        ("v1", "accepted"),
    ]
    assert payload["gaps"] == []

    result = client.get("/api/v1/batches/B1/result").json()
    assert result["complete"] is True
    assert result["revisions_total"] == 2
    assert payload["timeline_hash"] == result["timeline_hash"]
    segs = result["sensors"][0]["segments"]
    assert [(s["start_ns"], s["end_ns"], s["value"], s["revision_id"]) for s in segs] == [
        (0, 100, 1.0, "v1"),
        (100, 200, 3.5, "v2"),
        (200, 300, 1.0, "v1"),
    ]


def test_package_missing_external_parent_listed_as_gap(client):
    r = _post_package(client, _package("PKG-1", [
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
    ]))
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "accepted"
    assert payload["items"] == [
        {"sensor_id": "S1", "revision_id": "v2",
         "idempotency_key": "e2", "status": "accepted"}
    ]
    assert payload["gaps"] == [
        {"batch_id": "B1", "sensor_id": "S1", "missing_revision_id": "v1",
         "awaited_by": ["v2"], "referenced_from": "v2"}
    ]
    # The blocked revision is stored but not folded yet.
    result = client.get("/api/v1/batches/B1/result").json()
    assert result["revisions_blocked"] == 1
    assert result["complete"] is False


def test_package_gap_backfilled_by_second_package(client):
    _post_package(client, _package("PKG-1", [
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
    ]))
    r = _post_package(client, _package("PKG-2", [
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
    ]))
    payload = r.json()
    assert payload["status"] == "accepted"
    assert payload["gaps"] == []
    result = client.get("/api/v1/batches/B1/result").json()
    assert result["complete"] is True
    assert payload["timeline_hash"] == result["timeline_hash"]


def test_package_multi_sensor_and_withdrawal(client):
    body = _package("PKG-1", [
        _rev("w1", "v1", 120, 180, withdrawn=True, key="e3"),
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
        _rev("p1", "ROOT", 0, 100, value=1.5, sensor="S2", key="e4", unit="bar"),
    ])
    r = _post_package(client, body)
    assert r.status_code == 200
    assert [i["status"] for i in r.json()["items"]] == ["accepted"] * 3
    result = client.get("/api/v1/batches/B1/result").json()
    assert result["complete"] is True
    sensors = {s["sensor_id"]: s for s in result["sensors"]}
    assert [
        (s["start_ns"], s["end_ns"], s["state"])
        for s in sensors["S1"]["segments"]
    ] == [(0, 120, "calibrated"), (120, 180, "withdrawn"), (180, 300, "calibrated")]
    assert sensors["S2"]["segments"][0]["unit"] == "bar"


def test_package_per_item_duplicate_status_for_known_revisions(client):
    # v1 already stored via the single-event endpoint with the same key.
    _post_event(client, "v1", "ROOT", 0, 300, value=1.0, key="e1")
    r = _post_package(client, _package("PKG-1", [
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
    ]))
    payload = r.json()
    assert payload["status"] == "accepted"
    assert [(i["revision_id"], i["status"]) for i in payload["items"]] == [
        ("v1", "duplicate"),
        ("v2", "accepted"),
    ]
    assert client.get("/api/v1/batches/B1/result").json()["revisions_total"] == 2


def test_package_duplicate_item_registers_new_idempotency_key(client):
    _post_event(client, "v1", "ROOT", 0, 300, value=1.0, key="e1")
    r = _post_package(client, _package("PKG-1", [
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1-new"),
    ]))
    assert r.json()["items"][0]["status"] == "duplicate"
    # The fresh key was registered: replaying it alone is a duplicate too.
    dup = _post_event(client, "v1", "ROOT", 0, 300, value=1.0, key="e1-new")
    assert dup.json()["status"] == "duplicate"


# ---------------------------------------------------------------------------
# Whole-package rejection: nothing is written, no package record remains
# ---------------------------------------------------------------------------


def test_package_in_package_duplicate_revision_id_rejected(client):
    body = _package("PKG-1", [
        _rev("v1", "ROOT", 0, 100, value=1.0, key="e1"),
        _rev("v1", "ROOT", 0, 100, value=1.0, key="e2"),
    ])
    r = _post_package(client, body)
    assert r.status_code == 422
    assert client.get("/api/v1/batches/B1/result").status_code == 404


def test_package_in_package_duplicate_idempotency_key_rejected(client):
    body = _package("PKG-1", [
        _rev("v1", "ROOT", 0, 100, value=1.0, key="same"),
        _rev("v2", "v1", 100, 200, value=2.0, key="same"),
    ])
    r = _post_package(client, body)
    assert r.status_code == 422
    assert client.get("/api/v1/batches/B1/result").status_code == 404


def test_package_invalid_revision_rejects_whole_package(client):
    body = _package("PKG-1", [
        _rev("v1", "ROOT", 0, 100, value=1.0, key="e1"),
        _rev("v2", "v1", 200, 100, value=2.0, key="e2"),  # inverted interval
    ])
    r = _post_package(client, body)
    assert r.status_code == 422
    # Nothing from the package was written, not even the valid revision.
    assert client.get("/api/v1/batches/B1/result").status_code == 404


def test_package_empty_revision_list_rejected(client):
    r = _post_package(client, _package("PKG-1", []))
    assert r.status_code == 422


def test_package_batch_id_mismatch_rejected(client):
    r = _post_package(client, _package("PKG-1", [
        _rev("v1", "ROOT", 0, 100, value=1.0, key="e1"),
    ], batch="OTHER"), batch="B1")
    assert r.status_code == 422
    assert client.get("/api/v1/batches/B1/result").status_code == 404


def test_package_conflict_with_stored_revision_is_atomic(client):
    _post_event(client, "v1", "ROOT", 0, 300, value=1.0, key="e1")
    # Same revision id, different content -> whole package conflicts.
    r = _post_package(client, _package("PKG-1", [
        _rev("v1", "ROOT", 0, 300, value=9.9, key="e9"),
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
    ]))
    assert r.status_code == 409
    # The non-conflicting sibling revision was not written either.
    result = client.get("/api/v1/batches/B1/result").json()
    assert result["revisions_total"] == 1
    # The rejected package left no record: the same package id may be
    # resubmitted with corrected content.
    fixed = _post_package(client, _package("PKG-1", [
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
    ]))
    assert fixed.status_code == 200
    assert fixed.json()["status"] == "accepted"


def test_package_conflict_with_stored_idempotency_key_is_atomic(client):
    _post_event(client, "v1", "ROOT", 0, 300, value=1.0, key="e1")
    # Key e1 already belongs to a different payload.
    r = _post_package(client, _package("PKG-1", [
        _rev("v2", "v1", 100, 200, value=3.5, key="e1"),
        _rev("v3", "v2", 200, 250, value=4.0, key="e3"),
    ]))
    assert r.status_code == 409
    result = client.get("/api/v1/batches/B1/result").json()
    assert result["revisions_total"] == 1


# ---------------------------------------------------------------------------
# Package-level idempotent replay
# ---------------------------------------------------------------------------


def test_package_replay_returns_first_saved_result(client):
    body = _package("PKG-1", [
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
    ])
    first = _post_package(client, body).json()
    assert first["status"] == "accepted"
    assert [g["missing_revision_id"] for g in first["gaps"]] == ["v1"]

    # Later single submissions move the live batch on.
    _post_event(client, "v1", "ROOT", 0, 300, value=1.0, key="e1")
    live = client.get("/api/v1/batches/B1/result").json()
    assert live["complete"] is True
    assert live["timeline_hash"] != first["timeline_hash"]

    # The exact replay still returns the originally committed result.
    replay = _post_package(client, body).json()
    assert replay["status"] == "duplicate"
    assert replay["timeline_hash"] == first["timeline_hash"]
    assert replay["items"] == first["items"]
    assert replay["gaps"] == first["gaps"]


def test_package_replay_is_order_independent(client):
    revisions = [
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
    ]
    first = _post_package(client, _package("PKG-1", revisions)).json()
    reordered = _post_package(
        client, _package("PKG-1", list(reversed(revisions)))
    ).json()
    assert reordered["status"] == "duplicate"
    assert reordered["timeline_hash"] == first["timeline_hash"]


def test_package_same_id_different_content_conflicts(client):
    _post_package(client, _package("PKG-1", [
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
    ]))
    r = _post_package(client, _package("PKG-1", [
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
    ]))
    assert r.status_code == 409
    # The conflicting attempt wrote nothing.
    assert client.get("/api/v1/batches/B1/result").json()["revisions_total"] == 1


def test_package_ids_are_scoped_per_batch(client):
    body = _package("PKG-1", [_rev("v1", "ROOT", 0, 100, value=1.0, key="e1")])
    assert _post_package(client, body).json()["status"] == "accepted"
    # The same package id in another batch is an independent package.
    other = _post_package(client, _package("PKG-1", [
        _rev("v1", "ROOT", 0, 100, value=2.0, key="e1"),
    ], batch="B2"), batch="B2")
    assert other.json()["status"] == "accepted"


def test_reset_clears_packages(client):
    body = _package("PKG-1", [_rev("v1", "ROOT", 0, 100, value=1.0, key="e1")])
    assert _post_package(client, body).json()["status"] == "accepted"
    client.post("/admin/reset")
    again = _post_package(client, body).json()
    assert again["status"] == "accepted"


# ---------------------------------------------------------------------------
# Interplay with the pre-existing endpoints
# ---------------------------------------------------------------------------


def test_package_created_batch_supports_existing_endpoints(client):
    _post_package(client, _package("PKG-1", [
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
    ]))
    # Sample-point lookup on the package-folded timeline.
    sample = client.get(
        "/api/v1/batches/B1/sensors/S1/sample", params={"timestamp_ns": 150}
    ).json()
    assert sample["state"] == "calibrated"
    assert sample["value"] == 3.5
    assert sample["parent_chain"] == ["v2", "v1", "ROOT"]

    # Single-event preview still works on the same batch.
    preview = client.post(
        "/api/v1/batches/B1/events:preview",
        json={
            "batch_id": "B1", "sensor_id": "S1", "revision_id": "v3",
            "parent_id": "v2", "idempotency_key": "e3", "start_ns": 120,
            "end_ns": 180, "withdrawn": True,
        },
    ).json()
    assert preview["status"] == "submittable"

    # The complete batch can be sealed with the package-committed hash.
    timeline_hash = client.get("/api/v1/batches/B1/result").json()["timeline_hash"]
    sealed = client.post(
        "/api/v1/batches/B1/seals",
        json={"batch_id": "B1", "seal_id": "SEAL-1",
              "timeline_hash": timeline_hash},
    ).json()
    assert sealed["status"] == "sealed"
