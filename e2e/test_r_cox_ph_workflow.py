"""
End-to-end test: federated R Cox Proportional Hazards workflow.
"""
import re
import time
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page

PROJECT_NAME = "e2e-r-cox-ph-project"

TASKS_JSON = (
    '[{"seq": 1, "model": "RCoxProportionalHazards",'
    ' "config": {"total_round": 1, "current_round": 1}}]'
)

SUCCESS_TIMEOUT_S = 180


def _parse_project_ids(page: Page) -> tuple[int, int]:
    for link in page.locator('a[href*="projects/"]').all():
        href = link.get_attribute("href") or ""
        m = re.search(r"projects/(\d+)/(\d+)", href)
        if m:
            return int(m.group(1)), int(m.group(2))
    raise AssertionError("Could not find a project detail link")


def _upload_dataset_for_site(page, base, project_id, site_id, csv_path):
    page.goto(f"{base}/controller/projects/{project_id}/{site_id}/")
    page.wait_for_load_state("domcontentloaded")
    upload_btn = page.locator('[id^="openUploadModal"]')
    upload_btn.wait_for(state="visible", timeout=10_000)
    upload_btn.click()
    page.wait_for_timeout(400)
    page.evaluate("document.getElementById('fileupload').style.display = 'block'")
    page.locator("#fileupload").set_input_files(str(csv_path))
    page.locator("#uploadDataset").click(force=True)
    page.wait_for_timeout(2_000)


def _wait_for_success(page, base, project_id, site_id, timeout=SUCCESS_TIMEOUT_S):
    url = f"{base}/controller/projects/{project_id}/{site_id}/"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        page.goto(url)
        page.wait_for_load_state("domcontentloaded")
        if page.locator("td:has-text('Success')").count() > 0:
            return
        time.sleep(5)
    raise AssertionError(f"Run did not reach 'Success' within {timeout}s")


@pytest.fixture(scope="module")
def pages(browser: Browser, base_a, base_b, base_c):
    ctx_a = browser.new_context()
    ctx_b = browser.new_context()
    ctx_c = browser.new_context()
    yield ctx_a.new_page(), ctx_b.new_page(), ctx_c.new_page()
    ctx_a.close()
    ctx_b.close()
    ctx_c.close()


def test_r_cox_ph_workflow(pages, base_a, base_b, base_c, fixtures_dir):
    """Full federated R Cox PH workflow."""
    page_a, page_b, page_c = pages

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

    page_a.goto(f"{base_a}/controller/projects/new/")
    page_a.fill('[name="name"]', PROJECT_NAME)
    page_a.fill('textarea[name="description"]', "E2E R Cox PH test")
    page_a.fill('textarea[name="tasks"]', TASKS_JSON)
    page_a.click('[name="create_project"]')
    page_a.wait_for_url(f"{base_a}/controller/", timeout=15_000)
    project_id, site_a_id = _parse_project_ids(page_a)

    for page, base in [(page_b, base_b), (page_c, base_c)]:
        page.goto(f"{base}/controller/projects/join/")
        page.fill('[name="name"]', PROJECT_NAME)
        page.fill('textarea[name="notes"]', "E2E participant")
        page.click('[name="join"]')
        page.wait_for_load_state("domcontentloaded")

    _, site_b_id = _parse_project_ids(page_b)
    _, site_c_id = _parse_project_ids(page_c)

    page_a.goto(f"{base_a}/controller/projects/{project_id}/{site_a_id}/")
    page_a.wait_for_load_state("domcontentloaded")
    page_a.click("#startButton")
    page_a.wait_for_load_state("domcontentloaded")
    time.sleep(3)

    _upload_dataset_for_site(page_b, base_b, project_id, site_b_id,
                             fixtures_dir / "survival_site_b.csv")
    _upload_dataset_for_site(page_c, base_c, project_id, site_c_id,
                             fixtures_dir / "survival_site_c.csv")
    time.sleep(15)
    _upload_dataset_for_site(page_a, base_a, project_id, site_a_id,
                             fixtures_dir / "survival_site_a.csv")

    _wait_for_success(page_a, base_a, project_id, site_a_id)
    _wait_for_success(page_b, base_b, project_id, site_b_id)
    _wait_for_success(page_c, base_c, project_id, site_c_id)

    page_a.goto(
        f"{base_a}/controller/runs/detail/1/{project_id}/{site_a_id}/"
    )
    page_a.wait_for_load_state("domcontentloaded")
    assert "Success" in page_a.content()
