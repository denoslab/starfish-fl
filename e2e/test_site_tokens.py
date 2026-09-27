"""
SF-09 end to end: every site on its own token for a whole BabelBrainFno run.

Runs against the BabelBrain workbench profile and needs the docker CLI:

    cd workbench && make babelbrain-up && make babelbrain-tokens

The admin issues one enrolment code per site; each site enrols with its uid
and gets a token. The controllers restart with ROUTER_TOKEN, run 2 rounds,
and every token must have been used. The controllers go back to Basic auth
afterwards.
"""
import os
import subprocess
import time
from pathlib import Path

import pytest
import requests

ROUTER = "http://localhost:8000/starfish/api/v1"
AUTH = ("admin", "1234")
SITES = {
    "A": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "B": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    "C": "cccccccc-cccc-cccc-cccc-cccccccccccc",
}
CONTROLLERS = ["controller-{}{}".format(s, suffix) for s in "abc"
               for suffix in ("", "-scheduler", "-run-worker", "-processor-worker")]
TASKS = [{"seq": 1, "model": "BabelBrainFno", "config": {
    "total_round": 2, "current_round": 1,
    "data_source": {"type": "babelbrain_store", "bucket_hz": 250000}, "min_samples": 20}}]
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


def restart_controllers(tokens):
    env = dict(os.environ, **{"BB_TOKEN_" + k: v for k, v in tokens.items()})
    result = subprocess.run(COMPOSE + ["up", "-d", "--wait", *CONTROLLERS], env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]


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
def tokens():
    try:
        api("GET", "/sites/").raise_for_status()
    except requests.RequestException as e:
        pytest.skip("BabelBrain workbench not running: {}".format(e))
    issued = {}
    for key, uid in SITES.items():
        code = api("POST", "/enrolment-codes/",
                   json={"note": "e2e site " + key}).json()["code"]
        response = requests.post(ROUTER + "/sites/enrol/", timeout=30, json={
            "code": code, "uid": uid, "name": "babelbrain-site-" + key.lower(), "description": "d"})
        assert response.status_code == 201, response.text
        issued[key] = response.json()["token"]
    restart_controllers(issued)
    try:
        yield issued
    finally:
        restart_controllers({k: "" for k in SITES})


def test_a_whole_run_on_site_tokens(tokens):
    # Wait until every site heartbeats with its token
    def connected():
        sites = {s["uid"]: s["status"] for s in list_all("/sites/")}
        return all(sites.get(uid) == "Connected" for uid in SITES.values())

    wait_for(connected, 120, "all sites to heartbeat")
    site_ids = {s["uid"]: s["id"] for s in list_all("/sites/")}
    name = "bbfl-tokens-{}".format(int(time.time()))
    for key in ("A", "B", "C"):
        response = api("POST", "/projects/", json={"name": name, "description": "SF-09",
                                                   "site": site_ids[SITES[key]], "tasks": TASKS})
        assert response.status_code == 201, response.text
    project = next(p for p in list_all("/projects/") if p["name"] == name)
    assert api("POST", "/runs", json={"project": project["id"]}).ok
    batch = next(p for p in list_all("/projects/")
                 if p["id"] == project["id"])["batch"]

    def finished():
        runs = api("GET", "/runs/detail/", params={
            "batch": batch, "project": project["id"], "site_uid": SITES["A"]}).json()["runs"]
        return runs if all(r["status"] in ("Success", "Failed") for r in runs) else None

    runs = wait_for(finished, 900, "the run to finish on tokens")
    assert {r["status"] for r in runs} == {"Success"}

    used = {}
    for t in api("GET", "/site-tokens/").json():  # newest first
        if t["revoked_at"] is None:
            used.setdefault(t["site_uid"], t["last_used_at"])
    for key, uid in SITES.items():
        assert used.get(uid), "site {} never used its token".format(key)

    # A site's token cannot read another site's run files
    other = next(r for r in runs if str(r["site_uid"]) == SITES["B"])
    response = requests.get(ROUTER + "/runs-action/files/", timeout=30, params={
        "run": other["id"], "type": "mid_artifacts"},
        headers={"Authorization": "Token " + tokens["C"]})
    assert response.status_code == 403
