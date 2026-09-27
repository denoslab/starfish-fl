"""
End-to-end test: BabelBrainFno runs on local sample stores, no dataset upload.

Runs against the BabelBrain workbench profile, not the default e2e stack:

    cd workbench && make babelbrain-up && make babelbrain-e2e

Scenario, through the router API only
-------------------------------------
Step 1 - Register sites a, b and c
Step 2 - Site a creates a BabelBrainFno project; b and c join
Step 3 - Start a run. Nobody uploads a dataset
Step 4 - Every run leaves Standby on its own and all three rounds finish
Step 5 - The router holds one global model per round
Step 6 - Every site's uploaded log shows it read its own store and trained,
         and no log holds a store path or an agent message

Uses the stand-in model on CPU, SF-04.
"""
import io
import time
import zipfile

import pytest
import requests

ROUTER = "http://localhost:8000/starfish/api/v1"
AUTH = ("admin", "1234")
SITES = {
    "a": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "b": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    "c": "cccccccc-cccc-cccc-cccc-cccccccccccc",
}
TASKS = [{
    "seq": 1,
    "model": "BabelBrainFno",
    "config": {
        "total_round": 3,
        "current_round": 1,
        "data_source": {"type": "babelbrain_store", "bucket_hz": 250000},
        "min_samples": 20,
    },
}]
TOTAL_ROUNDS = 3
EXPECTED_FINAL_STATUS = "Success"
STORE_LINE = "Sample store at 250000 Hz: 20 train, 4 val"
TRAINED_LINE = "train samples, loss"
START_TIMEOUT_S = 90
FINISH_TIMEOUT_S = 900


def api(method, path, **kwargs):
    response = requests.request(
        method, ROUTER + path, auth=AUTH, timeout=30, **kwargs)
    return response


def list_all(path):
    """Every item of a paginated router list endpoint."""
    items, url = [], ROUTER + path
    while url:
        response = requests.get(url, auth=AUTH, timeout=30)
        response.raise_for_status()
        page = response.json()
        if isinstance(page, list):
            return page
        items.extend(page["results"])
        url = page["next"]
    return items


def ensure_site(name, uid):
    for site in list_all("/sites/"):
        if site["uid"] == uid:
            return site["id"]
    response = api("POST", "/sites/", json={
        "uid": uid, "name": "babelbrain-site-" + name, "description": "BabelBrain workbench"})
    assert response.status_code in (200, 201), response.text
    return ensure_site(name, uid)


def project_by_name(name):
    for project in list_all("/projects/"):
        if project["name"] == name:
            return project
    return None


def all_runs(project_id, batch):
    response = api("GET", "/runs/detail/", params={
        "batch": batch, "project": project_id, "site_uid": SITES["a"]})
    assert response.ok, response.text
    return response.json()["runs"]


@pytest.fixture(scope="module")
def started_run():
    try:
        api("GET", "/sites/").raise_for_status()
    except requests.RequestException as e:
        pytest.skip("BabelBrain workbench not running: {}".format(e))

    site_ids = {name: ensure_site(name, uid) for name, uid in SITES.items()}
    project_name = "bbfl-e2e-{}".format(int(time.time()))
    for name in ("a", "b", "c"):
        response = api("POST", "/projects/", json={
            "name": project_name, "description": "BabelBrain store e2e",
            "site": site_ids[name], "tasks": TASKS})
        assert response.status_code == 201, response.text
    project = project_by_name(project_name)
    assert project is not None

    response = api("POST", "/runs", json={"project": project["id"]})
    assert response.ok, response.text
    batch = project_by_name(project_name)["batch"]
    return project["id"], batch


def wait_for(predicate, timeout_s, what):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(2)
    raise AssertionError(
        "Timed out waiting for {}; last state {}".format(what, last))


def test_runs_start_without_dataset_upload(started_run):
    project_id, batch = started_run
    runs = all_runs(project_id, batch)
    assert len(runs) == 3

    def left_standby():
        statuses = [r["status"] for r in all_runs(project_id, batch)]
        return statuses if all(s != "Standby" for s in statuses) else None

    wait_for(left_standby, START_TIMEOUT_S, "all runs to leave Standby")


def test_runs_finish_and_logs_show_each_store(started_run):
    project_id, batch = started_run

    def finished():
        runs = all_runs(project_id, batch)
        done = all(r["status"] in ("Success", "Failed") for r in runs)
        return runs if done else None

    runs = wait_for(finished, FINISH_TIMEOUT_S, "all runs to finish")
    assert {r["status"] for r in runs} == {EXPECTED_FINAL_STATUS}

    coordinator = next(r for r in runs if r["role"] in (
        "CO", "Coordinator", "coordinator"))

    def logs_uploaded():
        response = api("GET", "/runs-action/download/", params={
            "run": coordinator["id"], "all_runs": "1", "type": "logs",
            "task_seq": 1, "round_seq": TOTAL_ROUNDS})
        if response.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            texts = [zf.read(n).decode("utf-8", "replace")
                     for n in zf.namelist()]
        return texts if len(texts) >= 3 else None

    texts = wait_for(logs_uploaded, 60, "logs from all three sites")
    combined = "\n".join(texts)
    assert sum(STORE_LINE in t for t in texts) == 3, combined[-2000:]
    assert sum(TRAINED_LINE in t for t in texts) == 3, combined[-2000:]
    assert "/babelbrain-store" not in combined
    assert "[Agent]" not in combined


def test_router_holds_one_global_model_per_round(started_run):
    project_id, batch = started_run
    runs = all_runs(project_id, batch)
    coordinator = next(r for r in runs if r["role"] in (
        "CO", "Coordinator", "coordinator"))
    response = api("GET", "/runs-action/files/", params={
        "run": coordinator["id"], "type": "artifacts", "all_runs": "1"})
    assert response.ok, response.text
    names = sorted(f["name"] for f in response.json())
    assert len(names) == TOTAL_ROUNDS, names
    assert all(f["sha256"] and f["size"] > 0 for f in response.json())
