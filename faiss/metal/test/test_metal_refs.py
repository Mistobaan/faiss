from __future__ import absolute_import, division, print_function, unicode_literals

import gc
import platform
import unittest

import faiss
import numpy as np


METAL_AVAILABLE = (
    platform.system() == "Darwin"
    and platform.machine() in ("arm64", "aarch64")
    and "METAL" in faiss.get_compile_options().split()
)


@unittest.skipUnless(METAL_AVAILABLE, "Metal backend not available")
class TestMetalReferences(unittest.TestCase):
    def test_constructor_keeps_resources_referenced(self):
        xb = np.random.RandomState(5).rand(128, 16).astype("float32")
        xq = xb[:8].copy()

        index = faiss.MetalIndexFlatL2(faiss.StandardMetalResources(), 16)
        self.assertTrue(hasattr(index, "referenced_objects"))
        self.assertEqual(len(index.referenced_objects), 1)
        self.assertIsInstance(index.referenced_objects[0], faiss.StandardMetalResources)

        gc.collect()
        index.add(xb)
        index.search(xq, 4)

    def test_index_cpu_to_metal_keeps_resources_referenced(self):
        xb = np.random.RandomState(9).rand(128, 20).astype("float32")
        xq = xb[:8].copy()

        cpu = faiss.IndexFlatL2(20)
        cpu.add(xb)

        index = faiss.index_cpu_to_metal(
            faiss.StandardMetalResources(),
            0,
            cpu,
        )
        self.assertTrue(hasattr(index, "referenced_objects"))
        self.assertEqual(len(index.referenced_objects), 1)
        self.assertIsInstance(index.referenced_objects[0], faiss.StandardMetalResources)

        gc.collect()
        D, I = index.search(xq, 4)
        self.assertEqual(D.shape, (8, 4))
        self.assertEqual(I.shape, (8, 4))
