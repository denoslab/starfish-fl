"""
SF-10 acceptance: three sites, one down for a round, quorum 2.

Runs against the BabelBrain workbench profile and needs the docker CLI:

    cd workbench && make babelbrain-up && make babelbrain-partial

The run starts with all three sites, and site c is stopped at once, mid
round 1. A stopped controller reports itself disconnected, so the run has
to start first. Round 1 waits for the deadline, then goes on with sites a
and b while c sits out. Site c is started again, rejoins round 2, and the
run ends in Success with c's delta in round 2.
"""
import json
import struct
import subprocess
import time
from pathlib import Path

import pytest
import requests

ROUTER = "http://localhost:8000/starfish/api/v1"
AUTH = ("admin", "1234")
SITES = {
    "a": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "b": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    "c": "cccccccc-cccc-cccc-cccc-cccccccccccc",
}
SITE_C_SERVICES = ["controller-c", "controller-c-scheduler", "controller-c-run-worker",
                   "controller-c-processor-worker"]
DEADLINE_MINUTES = 1.5
TASKS = [{"seq": 1, "model": "BabelBrainFno", "config": {
    "total_round": 2, "current_round": 1,
    "data_source": {"type": "babelbrain_store", "bucket_hz": 250000},
    "min_samples": 20, "min_participants": 2, "round_deadline_minutes": DEADLINE_MINUTES}}]
COMPOSE = ["docker", "compose", "-p", "starfish-babelbrain", "-f",
           str(Path(__file__).resolve().parent.parent / "workbench" / "compose" / "babelbrain.yaml")]


def api(method, path, **kwargs):
    return requests.request(method, ROUTER + path, auth=AUTH, timeout=30, **kwargs)


def list_all(path):
    items, url = [], ROUTER + path
    while url:
        page = requests.get(url, auth=AUTH, timeout=30).json()
        items.extend(page["results"])
        url = page["next"]
    return items


def compose(*args):
    result = subprocess.run(COMPOSE + list(args),
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]


def starfish_meta(data):
    """The Starfish metadata of a safetensors file, read from its header."""
    (length,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + length])
    return json.loads(header["__metadata__"]["starfish_meta"])


def wait_for(predicate, timeout_s, what):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(3)
    raise AssertionError(
        "Timed out waiting for {}; last state {}".format(what, last))


@pytest.fixture(scope="module")
def run_with_site_c_down():
    try:
        api("GET", "/sites/").raise_for_status()
    except requests.RequestException as e:
        pytest.skip("BabelBrain workbench not running: {}".format(e))
    sites = {s["uid"]: s["id"] for s in list_all("/sites/")}
    missing = [name for name, uid in SITES.items() if uid not in sites]
    if missing:
        pytest.skip(
            "run test_babelbrain_workflow.py first to register sites {}".format(missing))

    name = "bbfl-partial-{}".format(int(time.time()))
    for key in ("a", "b", "c"):
        response = api("POST", "/projects/", json={
            "name": name, "description": "SF-10 partial participation",
            "site": sites[SITES[key]], "tasks": TASKS})
        assert response.status_code == 201, response.text
    project = next(p for p in list_all("/projects/") if p["name"] == name)

    response = api("POST", "/runs", json={"project": project["id"]})
    assert response.ok, response.text
    compose("stop", *SITE_C_SERVICES)
    try:
        batch = next(p for p in list_all("/projects/")
                     if p["id"] == project["id"])["batch"]
        yield project["id"], batch
    finally:
        compose("start", *SITE_C_SERVICES)


def runs_of(project_id, batch):
    return api("GET", "/runs/detail/", params={
        "batch": batch, "project": project_id, "site_uid": SITES["a"]}).json()["runs"]


def test_round_completes_without_site_c_and_c_rejoins(run_with_site_c_down):
    project_id, batch = run_with_site_c_down
    seen_c = set()

    def round_two_started():
        runs = runs_of(project_id, batch)
        c = next(r for r in runs if str(r["site_uid"]) == SITES["c"])
        seen_c.add(c["status"])
        rounds = {r["tasks"][0]["config"]["current_round"] for r in runs}
        return runs if rounds == {2} else None

    wait_for(round_two_started, 600, "round 1 to finish without site c")
    assert "Sitting Out" in seen_c, seen_c

    compose("start", *SITE_C_SERVICES)

    def finished():
        runs = runs_of(project_id, batch)
        done = all(r["status"] in ("Success", "Failed") for r in runs)
        return runs if done else None

    runs = wait_for(finished, 900, "round 2 to finish with site c back")
    assert {r["status"] for r in runs} == {"Success"}, [
        (r["site_uid"], r["status"]) for r in runs]

    coordinator = next(r for r in runs if r["role"] == "coordinator")
    sites_per_round = {}
    for entry in api("GET", "/runs-action/files/", params={
            "run": coordinator["id"], "type": "artifacts", "all_runs": "1"}).json():
        data = requests.get(ROUTER + "/runs-action/file/", auth=AUTH, timeout=60, params={
            "run": coordinator["id"], "type": "artifacts", "name": entry["name"],
            "all_runs": "1"}).content
        meta = starfish_meta(data)
        sites_per_round[meta["round"]] = meta["metrics"]["sites"]
    assert sites_per_round == {1: 2, 2: 3}, sites_per_round
