"""
SF-02 part A acceptance: a 2 GB file through the workbench with bounded memory.

Runs against the BabelBrain workbench profile:

    cd workbench && make babelbrain-up && make babelbrain-transfer

A controller container uploads a random file to the router through the
streamed endpoint and downloads it back. The test checks the hashes and that
peak RSS stays under 500 MB in both the controller process and the router's
server processes. Needs the docker CLI; the size can be set with
STARFISH_TRANSFER_CHECK_MB.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import requests

ROUTER = "http://localhost:8000/starfish/api/v1"
AUTH = ("admin", "1234")
SITE_UID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
SIZE_MB = int(os.getenv("STARFISH_TRANSFER_CHECK_MB", "2048"))
MAX_RSS_MB = 500
COMPOSE = ["docker", "compose", "-p", "starfish-babelbrain", "-f",
           str(Path(__file__).resolve().parent.parent / "workbench" / "compose" / "babelbrain.yaml")]


def api(method, path, **kwargs):
    return requests.request(method, ROUTER + path, auth=AUTH, timeout=30, **kwargs)


def list_all(path):
    """Every item of a paginated router list endpoint."""
    items, url = [], ROUTER + path
    while url:
        page = requests.get(url, auth=AUTH, timeout=30).json()
        items.extend(page["results"])
        url = page["next"]
    return items


def compose(*args, timeout=1800):
    return subprocess.run(COMPOSE + list(args), capture_output=True, text=True, timeout=timeout)


def router_peak_rss_mb():
    """Highest VmHWM over the router container's python processes."""
    script = ("for s in /proc/[0-9]*/status; do "
              "grep -q '^Name:.*python' $s 2>/dev/null && grep '^VmHWM' $s; done")
    out = compose("exec", "-T", "router", "sh",
                  "-c", script, timeout=60).stdout
    values = [int(line.split()[1])
              for line in out.splitlines() if line.startswith("VmHWM")]
    assert values, "no python process found in the router container"
    return max(values) / 1024


@pytest.fixture(scope="module")
def run_id():
    try:
        api("GET", "/sites/").raise_for_status()
    except requests.RequestException as e:
        pytest.skip("BabelBrain workbench not running: {}".format(e))
    site = next((s for s in list_all("/sites/") if s["uid"] == SITE_UID), None)
    if site is None:
        response = api("POST", "/sites/", json={
            "uid": SITE_UID, "name": "babelbrain-site-a", "description": "BabelBrain workbench"})
        assert response.status_code in (200, 201), response.text
        site = next(s for s in list_all("/sites/") if s["uid"] == SITE_UID)
    # A single-site project whose task waits for a dataset, so the run just sits in Standby
    name = "bbfl-transfer-{}".format(int(time.time()))
    response = api("POST", "/projects/", json={
        "name": name, "description": "SF-02 transfer check", "site": site["id"],
        "tasks": [{"seq": 1, "model": "LogisticRegression",
                   "config": {"total_round": 1, "current_round": 1}}]})
    assert response.status_code == 201, response.text
    project = next(p for p in list_all("/projects/") if p["name"] == name)
    response = api("POST", "/runs", json={"project": project["id"]})
    assert response.ok, response.text
    batch = next(p for p in list_all("/projects/")
                 if p["id"] == project["id"])["batch"]
    runs = api("GET", "/runs/detail/", params={
        "batch": batch, "project": project["id"], "site_uid": SITE_UID}).json()["runs"]
    return runs[0]["id"]


def test_large_file_round_trip_with_bounded_memory(run_id):
    before = router_peak_rss_mb()
    result = compose("exec", "-T", "controller-a-processor-worker", "python3", "-m",
                     "starfish.controller.file.transfer_check", "--run", str(
                         run_id),
                     "--size-mb", str(SIZE_MB), "--folder", "/starfish-controller/local")
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    after = router_peak_rss_mb()
    print("transfer report:", report,
          "router peak RSS MB before/after:", before, after)

    stored = "/starfish/artifacts/{}/1/1/{}".format(
        run_id, report["router_name"])
    compose("exec", "-T", "router", "rm", "-f", stored, timeout=60)

    assert report["match"]
    assert report["bytes"] == SIZE_MB * 1024 * 1024
    assert report["peak_rss_mb"] < MAX_RSS_MB
    assert after < MAX_RSS_MB
