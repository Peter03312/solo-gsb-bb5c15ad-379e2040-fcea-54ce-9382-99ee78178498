"""End-to-end API behavior for atomic multi-revision package uploads.

An offline collection device uploads every revision of one acquisition run
as a single package: parents and children may be listed in any order, the
package commits atomically (or not at all), and an exact retry under the
same package id returns the first-saved result.
"""


def _rev(rev, parent, start, end, value=None, withdrawn=False, key=None,
         sensor="S1", unit="mm"):
    return {
        "sensor_id": sensor,
        "revision_id": rev,
        "parent_id": parent,
        "idempotency_key": key or f"k-{sensor}-{rev}",
        "start_ns": start,
        "end_ns": end,
        "value": value,
        "unit": unit,
        "withdrawn": withdrawn,
    }


def _package(*revs, batch="B1", package_id="PKG-1"):
    return {
        "batch_id": batch,
        "package_id": package_id,
        "revisions": list(revs),
    }


def _post_package(client, body, batch="B1"):
    return client.post(f"/api/v1/batches/{batch}/packages", json=body)


def _post_event(client, rev, parent, start, end, value=None, withdrawn=False,
                key=None, batch="B1", sensor="S1", unit="mm"):
    body = {
        "batch_id": batch,
        "sensor_id": sensor,
        "revision_id": rev,
        "parent_id": parent,
        "idempotency_key": key or f"k-{sensor}-{rev}",
        "start_ns": start,
        "end_ns": end,
        "value": value,
        "unit": unit,
        "withdrawn": withdrawn,
    }
    return client.post(f"/api/v1/batches/{batch}/events", json=body)


# ---------------------------------------------------------------------------
# Happy path: out-of-order package, merged causality, per-revision statuses
# ---------------------------------------------------------------------------


def test_package_out_of_order_parent_child_folded_after_merge(client):
    # Child listed before its parent inside the same package.
    body = _package(
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
    )
    r = _post_package(client, body)
    assert r.status_code == 200
    resp = r.json()
    assert resp["status"] == "accepted"
    assert resp["batch_id"] == "B1" and resp["package_id"] == "PKG-1"
    # Per-revision statuses echo the request order.
    assert [(x["revision_id"], x["status"]) for x in resp["revisions"]] == [
        ("v2", "accepted"),
        ("v1", "accepted"),
    ]
    result = resp["result"]
    # The whole package is merged before folding: no gap for the in-package
    # parent, both revisions grounded, deterministic fold applied.
    assert result["gaps"] == []
    assert result["revisions_grounded"] == 2
    assert result["complete"] is True
    segs = result["sensors"][0]["segments"]
    assert [(s["start_ns"], s["end_ns"], s["value"]) for s in segs] == [
        (0, 100, 1.0),
        (100, 200, 3.5),
        (200, 300, 1.0),
    ]
    # The committed result matches a subsequent live query.
    assert client.get("/api/v1/batches/B1/result").json() == result


def test_package_missing_external_parent_listed_as_gap(client):
    # The parent is neither stored nor inside the package: the package still
    # commits and the missing parent is reported as a backfillable gap.
    r = _post_package(client, _package(_rev("v2", "v1", 100, 200, value=3.5)))
    assert r.status_code == 200
    resp = r.json()
    assert resp["status"] == "accepted"
    result = resp["result"]
    assert result["revisions_blocked"] == 1
    assert result["complete"] is False
    assert result["gaps"] == [
        {
            "batch_id": "B1",
            "sensor_id": "S1",
            "missing_revision_id": "v1",
            "awaited_by": ["v2"],
            "referenced_from": "v2",
        }
    ]
    # Backfilling the gap afterwards with a single submission activates it.
    r2 = _post_event(client, "v1", "ROOT", 0, 300, value=1.0, key="e1")
    assert r2.json()["result"]["complete"] is True


def test_package_multi_sensor_and_withdrawn_tombstone(client):
    body = _package(
        _rev("w1", "v2", 120, 180, withdrawn=True, key="w1"),
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
        _rev("x1", "ROOT", 0, 50, value=9.0, key="kx", sensor="S2", unit="bar"),
    )
    resp = _post_package(client, body).json()
    assert resp["status"] == "accepted"
    assert all(x["status"] == "accepted" for x in resp["revisions"])
    result = resp["result"]
    assert result["complete"] is True
    sensors = {s["sensor_id"]: s for s in result["sensors"]}
    s1 = [(s["start_ns"], s["end_ns"], s["state"], s["value"])
          for s in sensors["S1"]["segments"]]
    assert s1 == [
        (0, 100, "calibrated", 1.0),
        (100, 120, "calibrated", 3.5),
        (120, 180, "withdrawn", None),
        (180, 200, "calibrated", 3.5),
        (200, 300, "calibrated", 1.0),
    ]
    assert [(s["start_ns"], s["end_ns"], s["value"])
            for s in sensors["S2"]["segments"]] == [(0, 50, 9.0)]


def test_package_response_carries_hash_and_gaps(client):
    resp = _post_package(
        client, _package(_rev("v1", "ROOT", 0, 100, value=1.0))
    ).json()
    assert len(resp["result"]["timeline_hash"]) == 64
    assert resp["result"]["gaps"] == []
    live = client.get("/api/v1/batches/B1/result").json()
    assert resp["result"]["timeline_hash"] == live["timeline_hash"]


# ---------------------------------------------------------------------------
# Whole-package rejection: nothing is written
# ---------------------------------------------------------------------------


def test_package_duplicate_revision_id_inside_rejected(client):
    body = _package(
        _rev("v1", "ROOT", 0, 100, value=1.0, key="k1"),
        _rev("v1", "ROOT", 0, 100, value=1.0, key="k2"),
    )
    r = _post_package(client, body)
    assert r.status_code == 422
    assert client.get("/api/v1/batches/B1/result").status_code == 404


def test_package_duplicate_idempotency_key_inside_rejected(client):
    body = _package(
        _rev("v1", "ROOT", 0, 100, value=1.0, key="same"),
        _rev("v2", "v1", 100, 200, value=2.0, key="same"),
    )
    r = _post_package(client, body)
    assert r.status_code == 422
    assert client.get("/api/v1/batches/B1/result").status_code == 404


def test_package_invalid_revision_rejected_and_nothing_written(client):
    # end_ns <= start_ns in one revision rejects the whole package.
    body = _package(
        _rev("v1", "ROOT", 0, 100, value=1.0, key="k1"),
        _rev("v2", "v1", 200, 150, value=2.0, key="k2"),
    )
    assert _post_package(client, body).status_code == 422
    # A withdrawn revision without a value is fine, but a non-withdrawn one
    # without a value is invalid.
    body = _package(
        _rev("v1", "ROOT", 0, 100, key="k1"),
    )
    assert _post_package(client, body).status_code == 422
    # The reserved revision id is invalid anywhere in the package.
    body = _package(_rev("ROOT", "ROOT", 0, 100, value=1.0, key="k1"))
    assert _post_package(client, body).status_code == 422
    # Nothing was written by any of the rejected packages.
    assert client.get("/api/v1/batches/B1/result").status_code == 404


def test_package_conflicting_with_stored_revision_rejected(client):
    _post_event(client, "v1", "ROOT", 0, 300, value=1.0, key="e1")
    # Same revision id, different content -> the whole package conflicts.
    body = _package(
        _rev("v1", "ROOT", 0, 300, value=9.9, key="new-key"),
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
    )
    r = _post_package(client, body)
    assert r.status_code == 409
    # Nothing of the package was written: v2 is unknown and the new key is
    # still free.
    result = client.get("/api/v1/batches/B1/result").json()
    assert result["revisions_total"] == 1
    r2 = _post_event(client, "v2", "v1", 100, 200, value=3.5, key="new-key")
    assert r2.json()["status"] == "accepted"


def test_package_conflicting_with_stored_idempotency_key_rejected(client):
    _post_event(client, "v1", "ROOT", 0, 300, value=1.0, key="e1")
    # Key e1 already used with a different payload -> whole package 409.
    body = _package(
        _rev("v9", "ROOT", 0, 50, value=2.0, key="e1"),
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
    )
    assert _post_package(client, body).status_code == 409
    result = client.get("/api/v1/batches/B1/result").json()
    assert result["revisions_total"] == 1


def test_package_partial_duplicate_reports_per_revision_status(client):
    # v1 was submitted on its own before the device uploaded its package.
    _post_event(client, "v1", "ROOT", 0, 300, value=1.0, key="e1")
    body = _package(
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
        # Identical content under a new key: a duplicate, and the new key is
        # registered exactly as in the single-event ingest.
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1-alt"),
    )
    resp = _post_package(client, body).json()
    assert resp["status"] == "accepted"
    assert [(x["revision_id"], x["status"]) for x in resp["revisions"]] == [
        ("v2", "accepted"),
        ("v1", "duplicate"),
    ]
    assert resp["result"]["revisions_total"] == 2
    # The newly registered key deduplicates single submissions going forward.
    r = _post_event(client, "v1", "ROOT", 0, 300, value=1.0, key="e1-alt")
    assert r.json()["status"] == "duplicate"


# ---------------------------------------------------------------------------
# Package-level idempotent replay
# ---------------------------------------------------------------------------


def test_package_retry_returns_first_saved_result_even_after_drift(client):
    body = _package(
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
    )
    first = _post_package(client, body).json()
    assert first["status"] == "accepted"

    # The live batch moves on after the package committed.
    _post_event(client, "v3", "v2", 30, 40, value=9.0, key="e3")

    # An exact retry returns the first-saved result verbatim.
    replay = _post_package(client, body)
    assert replay.status_code == 200
    assert replay.json()["status"] == "duplicate"
    assert replay.json()["revisions"] == first["revisions"]
    assert replay.json()["result"] == first["result"]
    # ...while the live batch has moved on.
    live = client.get("/api/v1/batches/B1/result").json()
    assert live["revisions_total"] == 3
    assert live["timeline_hash"] != first["result"]["timeline_hash"]


def test_package_retry_is_order_insensitive(client):
    revs = [
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
        _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
    ]
    first = _post_package(client, _package(*revs)).json()
    # Same content, different listing order: still an exact retry.
    replay = _post_package(client, _package(*reversed(revs)))
    assert replay.status_code == 200
    assert replay.json()["status"] == "duplicate"
    assert replay.json()["result"] == first["result"]


def test_package_same_id_different_content_conflicts(client):
    _post_package(client, _package(_rev("v1", "ROOT", 0, 300, value=1.0)))
    # Same package id, changed revision content -> 409.
    changed = _package(_rev("v1", "ROOT", 0, 300, value=2.0))
    assert _post_package(client, changed).status_code == 409
    # Same package id, extra revision -> 409.
    extended = _package(
        _rev("v1", "ROOT", 0, 300, value=1.0),
        _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
    )
    assert _post_package(client, extended).status_code == 409
    # Same package id, same revisions but renamed idempotency key -> 409.
    rekeyed = _package(_rev("v1", "ROOT", 0, 300, value=1.0, key="other"))
    assert _post_package(client, rekeyed).status_code == 409
    # The original package is still the one replaying correctly.
    replay = _post_package(
        client, _package(_rev("v1", "ROOT", 0, 300, value=1.0))
    )
    assert replay.json()["status"] == "duplicate"


def test_package_ids_are_scoped_per_batch(client):
    body_b1 = _package(_rev("v1", "ROOT", 0, 100, value=1.0), batch="B1")
    body_b2 = _package(_rev("v9", "ROOT", 0, 50, value=2.0), batch="B2")
    assert _post_package(client, body_b1, batch="B1").json()["status"] == "accepted"
    # The same package id in another batch is an independent package.
    r = _post_package(client, body_b2, batch="B2")
    assert r.json()["status"] == "accepted"


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_package_path_body_batch_mismatch_rejected(client):
    body = _package(_rev("v1", "ROOT", 0, 100, value=1.0), batch="B1")
    assert _post_package(client, body, batch="OTHER").status_code == 422


def test_package_empty_revisions_rejected(client):
    body = {"batch_id": "B1", "package_id": "PKG-1", "revisions": []}
    assert _post_package(client, body).status_code == 422


def test_package_missing_and_extra_fields_rejected(client):
    body = _package(_rev("v1", "ROOT", 0, 100, value=1.0))
    del body["package_id"]
    assert _post_package(client, body).status_code == 422
    body = _package(_rev("v1", "ROOT", 0, 100, value=1.0))
    body["unexpected"] = True
    assert _post_package(client, body).status_code == 422
    # Revisions carry no batch_id of their own (inherited from the package).
    body = _package(_rev("v1", "ROOT", 0, 100, value=1.0))
    body["revisions"][0]["batch_id"] = "B1"
    assert _post_package(client, body).status_code == 422


# ---------------------------------------------------------------------------
# Interaction with the rest of the service
# ---------------------------------------------------------------------------


def test_single_ingest_preview_and_seal_still_work_after_package(client):
    _post_package(
        client,
        _package(
            _rev("v2", "v1", 100, 200, value=3.5, key="e2"),
            _rev("v1", "ROOT", 0, 300, value=1.0, key="e1"),
        ),
    )
    # Single-event ingest on top of the package-committed batch.
    r = _post_event(client, "v3", "v2", 30, 40, value=2.5, key="e3")
    assert r.json()["status"] == "accepted"
    # Preview still works and does not write.
    preview = client.post(
        "/api/v1/batches/B1/events:preview",
        json={
            "batch_id": "B1", "sensor_id": "S1", "revision_id": "v4",
            "parent_id": "v3", "idempotency_key": "e4",
            "start_ns": 35, "end_ns": 45, "value": 4.0, "unit": "mm",
        },
    ).json()
    assert preview["status"] == "submittable"
    assert client.get("/api/v1/batches/B1/result").json()["revisions_total"] == 3
    # Sealing the completed batch still works.
    timeline_hash = client.get("/api/v1/batches/B1/result").json()["timeline_hash"]
    sealed = client.post(
        "/api/v1/batches/B1/seals",
        json={"batch_id": "B1", "seal_id": "SEAL-1",
              "timeline_hash": timeline_hash},
    ).json()
    assert sealed["status"] == "sealed"


def test_reset_clears_packages_and_counts(client):
    _post_package(client, _package(_rev("v1", "ROOT", 0, 100, value=1.0)))
    r = client.post("/admin/reset")
    assert r.json()["packages_removed"] == 1
    assert r.json()["events_removed"] == 1
    # After the reset the same package id commits as new again.
    resp = _post_package(client, _package(_rev("v1", "ROOT", 0, 100, value=1.0)))
    assert resp.json()["status"] == "accepted"
