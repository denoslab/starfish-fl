"""A site's own device setting wins over the project's."""

from unittest import TestCase, skipUnless

from starfish.controller.tasks.babel_brain_fno import training as T

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@skipUnless(HAS_TORCH, 'torch is not installed')
class PickDeviceTest(TestCase):

    def test_the_site_setting_wins_over_the_project_config(self):
        self.assertEqual(T.pick_device('cuda', environ={T.DEVICE_ENV: 'cpu'}).type, 'cpu')

    def test_the_project_config_applies_without_a_site_setting(self):
        self.assertEqual(T.pick_device('cpu', environ={}).type, 'cpu')
        self.assertEqual(T.pick_device('cpu', environ={T.DEVICE_ENV: '  '}).type, 'cpu')

    def test_auto_picks_an_available_device(self):
        device = T.pick_device('auto', environ={})
        self.assertIn(device.type, ('cuda', 'mps', 'cpu'))
        if device.type == 'cpu':
            self.assertFalse(torch.cuda.is_available())
