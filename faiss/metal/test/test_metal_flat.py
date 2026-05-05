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


def make_data(nb, nq, d, seed=123):
    rs = np.random.RandomState(seed)
    xb = rs.rand(nb, d).astype("float32")
    xq = rs.rand(nq, d).astype("float32")
    return xb, xq


@unittest.skipUnless(METAL_AVAILABLE, "Metal backend not available")
class TestMetalFlat(unittest.TestCase):
    def test_flat_l2_direct_and_copy_constructor(self):
        d = 32
        xb, xq = make_data(512, 32, d)
        k = 10

        cpu = faiss.IndexFlatL2(d)
        cpu.add(xb)
        D_cpu, I_cpu = cpu.search(xq, k)

        res = faiss.StandardMetalResources()
        index = faiss.MetalIndexFlatL2(res, d)
        index.add(xb)

        self.assertEqual(index.getDevice(), 0)
        self.assertEqual(index.getNumVecs(), xb.shape[0])

        D_metal, I_metal = index.search(xq, k)
        np.testing.assert_array_equal(I_metal, I_cpu)
        np.testing.assert_allclose(D_metal, D_cpu, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(index.reconstruct(7), xb[7], rtol=0, atol=0)

        copied = faiss.MetalIndexFlatL2(res, cpu)
        D_copy, I_copy = copied.search(xq, k)
        np.testing.assert_array_equal(I_copy, I_cpu)
        np.testing.assert_allclose(D_copy, D_cpu, rtol=1e-6, atol=1e-6)

    def test_flat_ip_direct_and_copy_constructor(self):
        d = 24
        xb, xq = make_data(384, 24, d, seed=321)
        k = 8

        cpu = faiss.IndexFlatIP(d)
        cpu.add(xb)
        D_cpu, I_cpu = cpu.search(xq, k)

        res = faiss.StandardMetalResources()
        index = faiss.MetalIndexFlatIP(res, d)
        index.add(xb)

        self.assertEqual(index.getNumVecs(), xb.shape[0])

        D_metal, I_metal = index.search(xq, k)
        np.testing.assert_array_equal(I_metal, I_cpu)
        np.testing.assert_allclose(D_metal, D_cpu, rtol=1e-6, atol=1e-6)

        copied = faiss.MetalIndexFlatIP(res, cpu)
        D_copy, I_copy = copied.search(xq, k)
        np.testing.assert_array_equal(I_copy, I_cpu)
        np.testing.assert_allclose(D_copy, D_cpu, rtol=1e-6, atol=1e-6)

    def test_index_cpu_to_metal_normalizes_generic_flat_indexes(self):
        d = 16
        xb, xq = make_data(256, 16, d, seed=777)

        flat_l2 = faiss.IndexFlat(d, faiss.METRIC_L2)
        flat_l2.add(xb)
        metal_l2 = faiss.index_cpu_to_metal(
            faiss.StandardMetalResources(), 0, flat_l2
        )
        self.assertIsInstance(metal_l2, faiss.MetalIndexFlatL2)

        D_cpu, I_cpu = flat_l2.search(xq, 5)
        D_metal, I_metal = metal_l2.search(xq, 5)
        np.testing.assert_array_equal(I_metal, I_cpu)
        np.testing.assert_allclose(D_metal, D_cpu, rtol=1e-6, atol=1e-6)

        flat_ip = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
        flat_ip.add(xb)
        metal_ip = faiss.index_cpu_to_metal(
            faiss.StandardMetalResources(), 0, flat_ip
        )
        self.assertIsInstance(metal_ip, faiss.MetalIndexFlatIP)

        D_cpu, I_cpu = flat_ip.search(xq, 5)
        D_metal, I_metal = metal_ip.search(xq, 5)
        np.testing.assert_array_equal(I_metal, I_cpu)
        np.testing.assert_allclose(D_metal, D_cpu, rtol=1e-6, atol=1e-6)
