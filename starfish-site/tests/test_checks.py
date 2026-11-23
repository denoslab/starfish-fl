"""Pieces of starfish-site check that need no router or GPU."""

import unittest

from starfish_site.checks import Report, arch_supported, check_settings

CU128 = ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120', 'compute_120']


class ArchTest(unittest.TestCase):

    def test_cards_of_the_testbed_run_on_a_cuda_12_8_build(self):
        self.assertTrue(arch_supported((8, 9), CU128))    # RTX 4070 Ti SUPER, Ada
        self.assertTrue(arch_supported((12, 0), CU128))   # RTX PRO 6000, Blackwell

    def test_a_build_without_blackwell_kernels_is_caught(self):
        self.assertFalse(arch_supported((12, 0), ['sm_80', 'sm_86', 'sm_90', 'compute_90a']))
        self.assertTrue(arch_supported((12, 0), ['sm_86', 'compute_90']))

    def test_older_minor_kernels_do_not_run_on_an_earlier_gpu(self):
        self.assertFalse(arch_supported((8, 0), ['sm_86']))
        self.assertTrue(arch_supported((9, 0), ['sm_90a']))
        self.assertFalse(arch_supported((10, 0), ['sm_90a']))


class SettingsTest(unittest.TestCase):

    def lines(self, values):
        out = []
        report = Report(out.append)
        check_settings(values, report)
        return report, '\n'.join(out)

    def test_plain_http_to_another_host_warns(self):
        report, text = self.lines({'SITE_UID': '8de14f02-de8a-4a8b-867a-8b5b9b7b92e3',
                                   'ROUTER_URL': 'http://10.0.0.5:8000/starfish/api/v1',
                                   'ROUTER_TOKEN': 't', 'BABELBRAIN_FL_STORE': '/'})
        self.assertEqual((report.failed, report.warned), (0, 1))
        self.assertIn('plain HTTP', text)

    def test_missing_enrolment_and_a_bad_device_fail(self):
        report, text = self.lines({'BABELBRAIN_FL_DEVICE': 'gpu'})
        self.assertEqual(report.failed, 4)
        self.assertIn('Enrol first', text)


if __name__ == '__main__':
    unittest.main()
