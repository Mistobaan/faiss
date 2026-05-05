from __future__ import absolute_import, division, print_function, unicode_literals

import platform
import unittest

import faiss
import numpy as np


METAL_AVAILABLE = (
    platform.system() == "Darwin"
    and platform.machine() in ("arm64", "aarch64")
    and "METAL" in faiss.get_compile_options().split()
)


def make_data(nb, nq, d, seed=1234):
    rs = np.random.RandomState(seed)
    xb = rs.rand(nb, d).astype("float32")
    xq = rs.rand(nq, d).astype("float32")
    return xb, xq


@unittest.skipUnless(METAL_AVAILABLE, "Metal backend not available")
class TestMetalTurboQuant(unittest.TestCase):
    def test_direct_construction_matches_cpu_reference(self):
        d = 32
        k = 6
        xb, xq = make_data(512, 24, d)

        cpu = faiss.IndexTurboQuantMSE(d, 2, faiss.METRIC_L2, 12345, True)
        cpu.add(xb)

        res = faiss.StandardMetalResources()
        metal = faiss.MetalIndexTurboQuantMSE(
            res, d, 2, faiss.METRIC_L2, 12345, True
        )
        metal.add(xb)

        D_cpu, I_cpu = cpu.search(xq, k)
        D_metal, I_metal = metal.search(xq, k)

        np.testing.assert_array_equal(I_metal, I_cpu)
        np.testing.assert_allclose(D_metal, D_cpu, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            metal.reconstruct(11), cpu.reconstruct(11), rtol=1e-4, atol=1e-4
        )

    def test_copy_constructor_and_clone_roundtrip(self):
        d = 48
        k = 5
        xb, xq = make_data(640, 20, d, seed=4321)

        cpu = faiss.IndexTurboQuantMSE(d, 4, faiss.METRIC_L2, 999, True)
        cpu.add(xb)

        res = faiss.StandardMetalResources()
        metal = faiss.MetalIndexTurboQuantMSE(res, cpu)
        self.assertEqual(metal.getDevice(), 0)
        self.assertEqual(metal.getNumVecs(), xb.shape[0])

        D_cpu, I_cpu = cpu.search(xq, k)
        D_metal, I_metal = metal.search(xq, k)
        np.testing.assert_array_equal(I_metal, I_cpu)
        np.testing.assert_allclose(D_metal, D_cpu, rtol=1e-4, atol=1e-4)

        roundtrip = faiss.index_metal_to_cpu(metal)
        D_rt, I_rt = roundtrip.search(xq, k)
        np.testing.assert_array_equal(I_rt, I_cpu)
        np.testing.assert_allclose(D_rt, D_cpu, rtol=1e-4, atol=1e-4)

        cloned = faiss.index_cpu_to_metal(res, 0, cpu)
        self.assertIsInstance(cloned, faiss.MetalIndexTurboQuantMSE)
        D_clone, I_clone = cloned.search(xq, k)
        np.testing.assert_array_equal(I_clone, I_cpu)
        np.testing.assert_allclose(D_clone, D_cpu, rtol=1e-4, atol=1e-4)
