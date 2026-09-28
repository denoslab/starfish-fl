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

## Commands

From `workbench/`, with Docker running and the `denoslab/BabelBrain` fork checked out next to `starfish-fl` on `feature/babelbrain-fl`. Set `BABELBRAIN_DIR` if it is somewhere else.

```bash
make babelbrain-demo-stores   # toy runs, then backfill and live export into the stores
make babelbrain-demo-up       # router and three sites on those stores
make babelbrain-demo-run      # one federated run of 3 rounds, narrated, about 2 minutes
make babelbrain-demo-tab      # optional: BabelBrain's FL tab on site B's store
make babelbrain-demo-down     # stop; make babelbrain-demo-clean also deletes the data
```

`make babelbrain-demo-run DEMO_ARGS="--peak-margin-mm 0.5 --rounds 4"` sets a gate margin and the number of rounds. The tab needs PySide6, numpy and h5py, for example BabelBrain's environment: `make babelbrain-demo-tab PYTHON=<that python>`.

The stores live in `workbench/.babelbrain-demo/`, which git ignores. Each site mounts only its own store, read-only. The demo runs as its own compose project, `starfish-babelbrain-demo`, and `make babelbrain-demo-up` stops the SF-06 profile, which uses the same ports.
