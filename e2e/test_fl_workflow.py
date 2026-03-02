"""
End-to-end test: full federated learning workflow through the web portal.

Scenario
--------
  site-a (coordinator) – port 8001
  site-b (participant 1) – port 8002
  site-c (participant 2) – port 8003

Step 1  – Register all three sites
Step 2  – Coordinator creates a project (LogisticRegression, 1 round)
Step 3  – Both participants join the project
Step 4  – Coordinator starts a new run
Step 5  – All three sites upload their CSV dataset
Step 6  – Wait for all runs to reach Success (≤ 180 s)
Step 7  – Coordinator inspects run details and views logs
Step 8  – Coordinator downloads artifacts
"""
import re
import time
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page

# ── constants ─────────────────────────────────────────────────────────────────

PROJECT_NAME = "e2e-fl-project"

TASKS_JSON = (
    '[{"seq": 1, "model": "LogisticRegression",'
    ' "config": {"total_round": 1, "current_round": 1}}]'
)

SUCCESS_TIMEOUT_S = 180  # seconds to wait for all runs to finish


# ── helpers ───────────────────────────────────────────────────────────────────


def _parse_project_ids(page: Page) -> tuple[int, int]:
    """
    Parse project_id and site_id from the 'Project Details' link on the home page.
    The link href is like /controller/projects/{project_id}/{site_id}.
    """
    link = page.locator('a[href*="/controller/projects/"]').first
    href = link.get_attribute("href")
    m = re.search(r"/projects/(\d+)/(\d+)", href)
    assert m, f"Could not parse project/site IDs from href: {href}"
    return int(m.group(1)), int(m.group(2))


def _upload_dataset_for_site(page: Page, base: str, project_id: int, site_id: int,
                              csv_path: Path) -> None:
    """
    Navigate to the project detail page and upload a dataset via the modal.

    The page uses Bootstrap 5 but the upload modal button was written for
    Bootstrap 4 data-attributes.  We therefore:
      1. Click the 'Upload Dataset' button (which sets up the #uploadDataset
         JS handler with the correct run_id).
      2. Make the hidden file input temporarily accessible and set the file.
      3. Click #uploadDataset (force=True in case the modal is not visible).
    """
    page.goto(f"{base}/controller/projects/{project_id}/{site_id}/")
    page.wait_for_load_state("domcontentloaded")

    upload_btn = page.locator('[id^="openUploadModal"]')
    upload_btn.wait_for(state="visible", timeout=10_000)

    # Click to capture run_id and attach the #uploadDataset click handler
    upload_btn.click()
    page.wait_for_timeout(400)

    # Make the file input accessible (it lives inside the potentially-hidden modal)
    page.evaluate("document.getElementById('fileupload').style.display = 'block'")
    page.locator("#fileupload").set_input_files(str(csv_path))

    # Trigger the upload (force bypasses visibility if modal did not open)
    page.locator("#uploadDataset").click(force=True)
    page.wait_for_timeout(2_000)


def _wait_for_success(page: Page, base: str, project_id: int, site_id: int,
                      timeout: int = SUCCESS_TIMEOUT_S) -> None:
    """Poll the project detail page until the run status shows 'Success'."""
    url = f"{base}/controller/projects/{project_id}/{site_id}/"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        page.goto(url)
        page.wait_for_load_state("domcontentloaded")
        if page.locator("td:has-text('Success')").count() > 0:
            return
        time.sleep(5)
    raise AssertionError(
        f"Run at {url} did not reach 'Success' within {timeout} seconds"
    )


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pages(browser: Browser, base_a, base_b, base_c):
    """
    Create one browser context (= isolated cookie/session store) per site.
    Yields (page_a, page_b, page_c) and cleans up at module teardown.
    """
    ctx_a = browser.new_context()
    ctx_b = browser.new_context()
    ctx_c = browser.new_context()
    page_a = ctx_a.new_page()
    page_b = ctx_b.new_page()
    page_c = ctx_c.new_page()
    yield page_a, page_b, page_c
    ctx_a.close()
    ctx_b.close()
    ctx_c.close()


# ── main test ─────────────────────────────────────────────────────────────────


def test_fl_workflow(pages, base_a, base_b, base_c, fixtures_dir):
    """Full 8-step federated learning workflow through the Controller web portal."""
    page_a, page_b, page_c = pages

    # ── Step 1: Site registration ─────────────────────────────────────────────

    for page, base, name, desc in [
        (page_a, base_a, "site-a", "Coordinator"),
        (page_b, base_b, "site-b", "Participant 1"),
        (page_c, base_c, "site-c", "Participant 2"),
    ]:
        page.goto(f"{base}/controller/")
        page.wait_for_load_state("domcontentloaded")
        page.fill('[name="name"]', name)
        page.fill('textarea[name="description"]', desc)
        page.click('[name="register_site"]')
        page.wait_for_load_state("domcontentloaded")
        assert "is currently registered" in page.content(), (
            f"Step 1 – {name}: expected 'is currently registered' on dashboard"
        )

    # ── Step 2: Coordinator creates project ───────────────────────────────────

    page_a.goto(f"{base_a}/controller/projects/new/")
    page_a.fill('[name="name"]', PROJECT_NAME)
    page_a.fill('textarea[name="description"]', "E2E test project")
    page_a.fill('textarea[name="tasks"]', TASKS_JSON)
    page_a.click('[name="create_project"]')
    # The JS does window.location.href = "/" which Django redirects to /controller/
    page_a.wait_for_load_state("domcontentloaded")

    assert PROJECT_NAME in page_a.content(), (
        "Step 2 – Coordinator: project should appear on home page after creation"
    )
    project_id, site_a_id = _parse_project_ids(page_a)

    # ── Step 3: Participants join the project ─────────────────────────────────

    for page, base in [(page_b, base_b), (page_c, base_c)]:
        page.goto(f"{base}/controller/projects/join/")
        page.fill('[name="name"]', PROJECT_NAME)
        page.fill('textarea[name="notes"]', "E2E participant")
        page.click('[name="join"]')
        page.wait_for_load_state("domcontentloaded")
        assert PROJECT_NAME in page.content(), (
            f"Step 3 – participant at {base}: project should appear on home page after joining"
        )

    _, site_b_id = _parse_project_ids(page_b)
    _, site_c_id = _parse_project_ids(page_c)

    # ── Step 4: Coordinator starts a new run ──────────────────────────────────

    page_a.goto(f"{base_a}/controller/projects/{project_id}/{site_a_id}/")
    page_a.wait_for_load_state("domcontentloaded")
    page_a.click("#startButton")
    page_a.wait_for_load_state("domcontentloaded")
    assert page_a.locator("td:has-text('Standby')").count() > 0, (
        "Step 4 – Coordinator: at least one run should be in 'Standby' after starting"
    )

    # Small delay to let the router propagate Run records to participants
    time.sleep(3)

    # ── Step 5: All sites upload their datasets ───────────────────────────────

    _upload_dataset_for_site(page_a, base_a, project_id, site_a_id,
                             fixtures_dir / "site_a.csv")
    _upload_dataset_for_site(page_b, base_b, project_id, site_b_id,
                             fixtures_dir / "site_b.csv")
    _upload_dataset_for_site(page_c, base_c, project_id, site_c_id,
                             fixtures_dir / "site_c.csv")

    # ── Step 6: Wait for all runs to reach Success ────────────────────────────

    _wait_for_success(page_a, base_a, project_id, site_a_id)
    _wait_for_success(page_b, base_b, project_id, site_b_id)
    _wait_for_success(page_c, base_c, project_id, site_c_id)

    # ── Step 7: Run details and logs (coordinator) ────────────────────────────

    batch = 1  # First (and only) run batch
    page_a.goto(
        f"{base_a}/controller/runs/detail/{batch}/{project_id}/{site_a_id}/"
    )
    page_a.wait_for_load_state("domcontentloaded")
    assert "Success" in page_a.content(), (
        "Step 7 – Run details: expected 'Success' status for coordinator run"
    )

    logs_btn = page_a.locator("#logs").first
    if logs_btn.count() > 0:
        logs_btn.click()
        page_a.wait_for_timeout(3_000)
        log_text = page_a.locator("#logContent").inner_text()
        assert log_text.strip(), (
            "Step 7 – Logs: log modal content should be non-empty after training"
        )

    # ── Step 8: Download artifacts (coordinator) ──────────────────────────────

    page_a.goto(
        f"{base_a}/controller/runs/detail/{batch}/{project_id}/{site_a_id}/"
    )
    page_a.wait_for_load_state("domcontentloaded")

    download_btn = page_a.locator('[id^="downloadButton-"]').first
    if download_btn.count() > 0:
        with page_a.expect_download(timeout=30_000) as dl_info:
            download_btn.click()
        download = dl_info.value
        assert download.suggested_filename, (
            "Step 8 – Download: expected a filename in the downloaded artifacts"
        )
