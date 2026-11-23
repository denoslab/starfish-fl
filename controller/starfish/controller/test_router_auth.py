"""The plain HTTP warning of router_auth comes once, also for threads that ask together."""

import threading
import time
from unittest import TestCase
from unittest.mock import patch

from starfish.controller import router_auth as auth


class PlainHttpWarningTest(TestCase):

    def setUp(self):
        for patcher in (patch.object(auth, '_warned_http', False),
                        patch.dict('os.environ', {'ROUTER_TOKEN': 'token',
                                                  'ROUTER_URL': 'http://127.0.0.1:18100'})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_threads_that_ask_together_get_one_warning(self):
        warnings = []

        def slow_warning(*args):
            # Holds the first caller long enough for the others to arrive
            time.sleep(0.05)
            warnings.append(args)

        start = threading.Barrier(8)
        answers = []

        def ask():
            start.wait()
            answers.append(auth.router_auth())

        with patch.object(auth.logger, 'warning', side_effect=slow_warning):
            threads = [threading.Thread(target=ask) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(len(warnings), 1)
        self.assertEqual([type(a) for a in answers], [auth.TokenAuth] * 8)

    def test_a_second_call_gives_no_warning(self):
        with patch.object(auth.logger, 'warning') as warning:
            auth.router_auth()
            auth.router_auth()
        warning.assert_called_once()

    def test_https_gives_no_warning(self):
        with patch.dict('os.environ', {'ROUTER_URL': 'https://router.lab'}), \
                patch.object(auth.logger, 'warning') as warning:
            auth.router_auth()
        warning.assert_not_called()
