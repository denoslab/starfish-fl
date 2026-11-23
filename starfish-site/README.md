# starfish-site

A native Starfish-FL site for Ubuntu and macOS. One process does the work of the controller's Celery beat, its two Celery workers and Redis, and it runs the controller's own task code. A site needs no Docker and no admin rights. The site trains on its own BabelBrain sample store and sends only weight updates to the router.

## What a site needs

- Ubuntu 22.04 or later on x86_64, or macOS on Apple silicon.
- On Ubuntu with an NVIDIA GPU, a driver of version 525 or later. Blackwell cards such as the RTX PRO 6000 need a driver that supports them, 570 or later.
- `curl` and `git`.
- About 8 GB of disk on Ubuntu with the CUDA build of torch, and less on macOS.
- From the router's admin, the router's URL, a single-use enrolment code, and the CA file when the router's certificate comes from the lab's own certificate authority.
- The site's BabelBrain sample store, readable by the user who runs the site.

## Install

```bash
git clone -b feature/babelbrain-fl https://github.com/denoslab/starfish-fl.git
cd starfish-fl
starfish-site/install.sh
```

The site runs the controller and starfish-site from this checkout. Moving or deleting the checkout breaks the site.

The script puts everything in `~/.starfish-site`. It downloads uv, a Python installer, and uses it to set up Python 3.11 in that folder, without touching the system Python. It then installs torch, as a build for CUDA 12.8 on Ubuntu with an NVIDIA driver and as the standard build with Apple's MPS on macOS, and the Starfish controller and starfish-site from the checkout. The `starfish-site` command is linked into `~/.local/bin`, and the script says how to add that folder to `PATH` when it is missing there.

| Option | Effect |
| --- | --- |
| `--cpu` | CPU-only torch on Ubuntu, a smaller download for a machine without a GPU |
| `--home DIR` | another site folder in place of `~/.starfish-site` |
| `--python PATH` | an existing Python 3.10 or 3.11 with venv and pip in place of uv |
| `--no-link` | no link in `~/.local/bin` |

## Enrol

The router's admin creates one enrolment code per site. The code works once and expires.

```bash
starfish-site enrol --router https://192.168.1.20:8443 --code <code> --name "Lab B" \
    --store /path/to/BabelBrainFL/samples --ca-cert router-ca.pem
```

Enrolment gives the site an ID and a token and writes both to `~/.starfish-site/site.env`, readable by the owner only. The token stands for the site in every request to the router, and a leaked token should be revoked by the admin. The coordinator also passes `--eval-store` with its held-out test set. A site whose automatic device choice fails passes `--device cpu`.

## Check

```bash
starfish-site check
```

The check reads the settings, asks the router who this site is, opens the sample stores, lists torch and the GPU, and trains a small model for one epoch on synthetic samples on the device the site will use. It compares that model's output with the CPU's. On a Mac this tells whether MPS can train the model. When it cannot, `BABELBRAIN_FL_DEVICE='cpu'` in `site.env` moves training to the CPU. Every line reads `ok`, `warn` or `FAIL`, with a hint under each problem. `--quick` skips the training part. Before enrolment the check covers torch and training only, which lets a practitioner test a machine before the admin issues a code.

## Run

```bash
starfish-site start
```

The site polls the router every 5 s, reports itself alive every 5 s and works on its runs one step at a time. It logs to the terminal and to `~/.starfish-site/logs/starfish-site.log`. Ctrl-C stops the site after the step in progress, and a second Ctrl-C stops it at once. `starfish-site stop` does the same from another terminal, and `starfish-site stop --now` cuts the step short. Only one process can run a site folder at a time. A stopped site shows as disconnected on the router and picks up its active runs when it starts again.

## The site folder

| Path | Holds |
| --- | --- |
| `site.env` | the settings, readable by the owner only |
| `router-ca.pem` | the router's CA file, when enrolment got one |
| `local/` | run files, logs and artifacts per run, task and round |
| `logs/` | `starfish-site.log`, rotated at 10 MB with 5 old files kept |
| `starfish-site.pid` | the running process, locked while it runs |
| `venv/`, `python/`, `bin/` | the installed software |

## Settings

`site.env` holds the controller's own environment variables, so the task code reads the same names as in a container. Enrolment writes the first ones, and the file is safe to edit while the site is stopped.

| Setting | Meaning |
| --- | --- |
| `SITE_UID` | the site's ID on the router |
| `ROUTER_URL` | the router API |
| `ROUTER_TOKEN` | the site's token from enrolment |
| `REQUESTS_CA_BUNDLE` | the CA file that signed the router's certificate |
| `BABELBRAIN_FL_STORE` | the site's sample store, read-only |
| `BABELBRAIN_FL_EVAL_STORE` | the coordinator's held-out test set for the evaluation gate |
| `BABELBRAIN_FL_DEVICE` | `cuda`, `mps` or `cpu`, which wins over the automatic choice and the project's device |
| `STARFISH_DISABLE_AGENTS` | `1` keeps all LLM agent code off, set by enrolment |

## Update and remove

After `git pull` the site runs the new code from its next start. When a pull changes dependencies, running `starfish-site/install.sh` again brings them in. Deleting `~/.starfish-site` and `~/.local/bin/starfish-site` removes a site, and `~/.cache/uv` holds uv's download cache.

## For developers

The polling and the task calls come from `controller/starfish/controller/run_dispatch.py`, which the controller's Celery tasks call as well. `starfish_site/runner.py` runs them on three timers and one processor thread, with the run cache in memory in place of Redis. The cache starts empty, as the controller's does after a restart, and a cached run expires after a day, as the Redis keys do. One difference from Celery is that a repeat of a message that already waits in the queue is dropped.

```bash
~/.starfish-site/venv/bin/python -m unittest discover -s starfish-site/tests
STARFISH_SITE=~/.starfish-site/venv/bin/starfish-site starfish-site/tests/e2e_native.sh <stores> 2 2
```

`e2e_native.sh` starts a test router in Docker on port 18100, enrols three sites into temporary folders, runs a project on the stores of `make babelbrain-demo-stores`, copied to be readable by the user, and removes everything at the end.
