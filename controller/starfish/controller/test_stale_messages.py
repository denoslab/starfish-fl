"""Stale Celery messages after a controller restart are dropped, SF-10 follow-up."""

from unittest import TestCase
from unittest.mock import MagicMock, patch

import requests

from starfish import celery as starfish_celery


def run(status='Standby', current_round=1, cur_seq=1):
    return {'id': 5, 'status': status, 'cur_seq': cur_seq,
            'tasks': [{'model': 'X', 'config': {'current_round': current_round}}]}


def answer(record, ok=True):
    response = MagicMock(ok=ok)
    response.json.return_value = record
    return response


class StaleMessageTest(TestCase):

    def check(self, message, current, ok=True):
        with patch.object(starfish_celery.requests, 'get', return_value=answer(current, ok)):
            return starfish_celery.is_stale(message)

    def test_current_message_is_kept(self):
        self.assertFalse(self.check(run(), run()))

    def test_message_for_an_earlier_round_is_stale(self):
        self.assertTrue(self.check(run(current_round=1), run(current_round=2)))

    def test_message_for_an_earlier_status_is_stale(self):
        self.assertTrue(self.check(run('Standby'), run('Preparing')))

    def test_message_is_kept_when_the_router_cannot_answer(self):
        self.assertFalse(self.check(run(), run(), ok=False))
        with patch.object(starfish_celery.requests, 'get',
                          side_effect=requests.ConnectionError('down')):
            self.assertFalse(starfish_celery.is_stale(run()))

    def test_stale_message_is_not_dispatched(self):
        with patch.object(starfish_celery, 'is_stale', return_value=True), \
                patch.object(starfish_celery, 'load_class') as load_class:
            starfish_celery.process_task.run(run(), False)
        load_class.assert_not_called()
