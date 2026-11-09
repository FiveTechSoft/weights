import unittest
import torch
from engine.closed_form_write import write_closed_form, fit_report


class ClosedFormWrite(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.W = torch.randn(48, 64, generator=g)
        self.K = torch.randn(64, 20, generator=g)
        self.R = torch.randn(48, 20, generator=g)
        self.P = torch.randn(64, 400, generator=g)
        self.Kn = torch.randn(64, 20, generator=g)

    def test_writes_associations(self):
        dW = write_closed_form(self.W, self.K, self.R, lam=1e-6)
        self.assertLess(fit_report(self.W, dW, self.K, self.R)["fit_resid"], 1e-3)

    def test_protection_reduces_drift(self):
        free = fit_report(self.W, write_closed_form(self.W, self.K, self.R, lam=1e-6), self.K, self.R, self.P)
        prot = fit_report(self.W, write_closed_form(self.W, self.K, self.R, self.P, lam=1e-6), self.K, self.R, self.P)
        self.assertLess(prot["protected_drift"], free["protected_drift"])

    def test_memorises_shuffled_targets_equally(self):
        # a write can fit ANY targets: fit quality on written keys says nothing about skill
        perm = torch.randperm(self.R.shape[1], generator=torch.Generator().manual_seed(1))
        dW = write_closed_form(self.W, self.K, self.R[:, perm], lam=1e-6)
        self.assertLess(fit_report(self.W, dW, self.K, self.R[:, perm])["fit_resid"], 1e-3)

    def test_unwritten_keys_do_not_get_the_target(self):
        dW = write_closed_form(self.W, self.K, self.R, lam=1e-6)
        # random new keys: the update does not reproduce the written outputs there
        self.assertGreater(float((dW @ self.Kn - self.R).norm() / self.R.norm()), 0.5)

    def test_shape_errors(self):
        with self.assertRaises(ValueError):
            write_closed_form(self.W, self.K[:10], self.R)


if __name__ == "__main__":
    unittest.main()
