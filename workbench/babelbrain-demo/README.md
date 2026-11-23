# BabelBrain FL demo

A demo on toy data that runs the whole path, from BabelBrain Step 2 outputs to an approved federated model. BabelBrain's own exporter and backfill tool write the sample stores. Three Starfish sites then train on them, and the coordinator's evaluation gate decides what reaches the model registry.

The meeting script, with talking points, is `runbooks/demo-synthetic-e2e.md` in `babelbrain-docs`.

## Real and stand-in parts

| Part | In the demo |
| --- | --- |
| Eligibility rules, store layout, group IDs, splits, counters, region labels, manifest | BabelBrain's code on the fork, unchanged |
| Backfill tool and the export call the Step 2 hook makes | BabelBrain's code, called from `make_stores.py` |
| Step 2 outputs | Toy files named like BabelBrain's, with no simulation in them |
| Crop and resampling | `toy_physics.demo_crop`, a stand-in until Tayeb's code is ported |
| Store reader, training, FedAvg, gate, per-region report, registry | Starfish, unchanged |
| Model | The stand-in FNO of SF-04, not tFUS-FNO |

The numbers show the mechanics only. They say nothing about real skulls or about tFUS-FNO.

The three labs' toy cases differ on purpose, in `SITE_CASES` of `make_stores.py`. A has thin skulls and B thick ones, tilted the same way, and C medium skulls tilted the other way. The NeuroFUS test set takes four subjects from each range, and no lab alone has seen that mix. Federation thus has something to add over training at one site. The local-only baseline of the live page makes that visible. In a run of 3 rounds of 4 local epochs the federated model reached 58.3 % relative l2 error on the test set, against 71.0 % for the local-only average and 43.5 % for the centralized baseline.

## Commands

From `workbench/`, with Docker running and the `denoslab/BabelBrain` fork checked out next to `starfish-fl` on `feature/babelbrain-fl`. Set `BABELBRAIN_DIR` if it is somewhere else.

```bash
make babelbrain-demo-stores   # toy runs, then backfill and live export into the stores
make babelbrain-demo-up       # router and three sites on those stores
make babelbrain-demo-run      # one federated run of 3 rounds, narrated, about 2 minutes
make babelbrain-demo-ui       # or the same run on a live page at http://localhost:8090
make babelbrain-demo-tab      # optional: BabelBrain's FL tab on site B's store
make babelbrain-demo-down     # stop; make babelbrain-demo-clean also deletes the data
```

`make babelbrain-demo-run DEMO_ARGS="--peak-margin-mm 0.5 --rounds 4"` adds an opt-in peak-distance margin to the gate, which otherwise decides on relative l2 alone, and sets the number of rounds. The tab needs PySide6, numpy and h5py, for example BabelBrain's environment: `make babelbrain-demo-tab PYTHON=<that python>`.

## Live page

`make babelbrain-demo-ui` serves `demo_ui.py`, the coordinator's console at site A, at http://localhost:8090. Its Start run button creates a project for sites A, B and C and starts a run with the given rounds, local epochs and optional peak-distance margin. The page then follows the run every two seconds. It shows each site's state per round and a narrated activity feed, the gate's verdict with relative l2 error, peak distance and SSIM per round, each site's training loss per epoch, and field slices from every global model on site A's held-out samples. The registry table lists the models the gate passed, and its Approve button signs off a pending model as the coordinator. A table view sits under each chart.

For comparison the page trains two baselines for the run it shows. Both call the task's own `train_round` with the same model, init:0 start, optimizer, loss, batch size and rounds of local epochs. The centralized baseline trains one model on the 70 train samples of sites A, B and C pooled. The local-only baseline trains one model per site on that site's own samples and never aggregates, and its scores are the simple average over the three sites. Every sample is thus seen as often as in the federated run. After each round every model is scored on site A's eval samples with the gate's metrics. In the gate and loss charts the centralized baseline is a solid grey line and the local-only average a dashed one. Both add a value to each tile and a column to the table of federated and baseline scores, a second table gives the local-only scores per site, and the field viewer offers the models `centralized`, `local A`, `local B` and `local C`. Each baseline trains in under 10 s, without a gate, and stays in memory for each run config. With all three stores on one machine, only this demo can train them. The baselines always train on the current stores. A run whose gate saw another test set, for example one from before the last `make babelbrain-demo-stores`, therefore shows them hidden. The relative l2 and peak distance charts scale to the rounds and show the far higher start as an arrow with its value.

The page runs in the controller image inside the compose network and talks only to the router's API, so the browser needs neither the router password nor CORS. For the field slices it reads site A's eval store, read-only, and loads the task's own stand-in model, training and metric code from the image. Its scores per model match the gate's report exactly. For the baselines it also reads the three site stores, read-only. It listens on localhost only and accepts its two actions from its own origin only. `make babelbrain-demo-down` stops it with the rest of the stack, and `docker logs starfish-babelbrain-demo-demo-ui-1` shows the same narration as the activity feed.

The stores live in `workbench/.babelbrain-demo/`, which git ignores. Rebuilding them deletes and writes the store folders again, and running containers keep the deleted ones. Rebuild with the stack down, or recreate the site containers and the page afterwards. Each site mounts only its own store, read-only. The demo runs as its own compose project, `starfish-babelbrain-demo`, and `make babelbrain-demo-up` stops the SF-06 profile, which uses the same ports.
