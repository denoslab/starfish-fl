"""
Open BabelBrain's Federated Learning tab on a demo store, without the rest of BabelBrain.

    python show_fl_tab.py --store .babelbrain-demo/stores/site-b [--babelbrain PATH]

The tab is the fork's ``FederatedLearning/SettingsWidget.py`` as it appears
in Advanced Options, here in a window of its own. Needs PySide6, numpy and
h5py, for example BabelBrain's own environment. In the full app, the same
tab shows this store after ``BABELBRAIN_FL_STORE=<store> python BabelBrain.py``.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--store', required=True)
    parser.add_argument('--babelbrain', default=os.environ.get('BABELBRAIN_DIR', os.path.join(
        HERE, '..', '..', '..', 'BabelBrain')), help='denoslab/BabelBrain checkout')
    parser.add_argument('--screenshot', help='save a PNG of the tab and exit')
    args = parser.parse_args(argv)
    sys.path.insert(0, os.path.join(
        os.path.abspath(args.babelbrain), 'BabelBrain'))

    from PySide6.QtWidgets import QApplication
    from FederatedLearning import FL_COLLECT
    from FederatedLearning.SettingsWidget import FLSettingsWidget

    app = QApplication.instance() or QApplication(sys.argv)
    widget = FLSettingsWidget(current_level=FL_COLLECT,
                              store_folder=os.path.abspath(args.store))
    widget.setWindowTitle(
        'BabelBrain, Advanced Options, Federated Learning (demo store)')
    widget.resize(760, 560)
    widget.show()
    if args.screenshot:
        app.processEvents()
        widget.grab().save(args.screenshot)
        return 0
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
