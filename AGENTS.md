# AGENTS.md — Notes for AI coding agents

## SWIG typemap ordering matters

**Rule:** Any `%include` of a header that declares functions returning
`faiss::Index *` (or `faiss::IndexBinary *`, etc.) **must appear after** the
corresponding `%typemap(out)` block in `faiss/python/swigfaiss.swig`.

SWIG applies output typemaps only to function declarations it encounters
*after* the `%typemap(out)` definition.  If a header is `%include`d before
the typemap, the generated wrapper will return the raw base-class pointer
(`faiss::Index *`) instead of downcasting to the concrete subclass via
`dynamic_cast`.  The symptom at the Python level is that
`isinstance(obj, faiss.SomeSubclass)` returns `False` even though the
underlying C++ object is of that type.

### Layout in `swigfaiss.swig`

```
1.  %include class headers         ← class definitions (needed for SWIGTYPE_p_…)
2.  %typemap(out) faiss::Index *   ← downcast logic (DOWNCAST / DOWNCAST_METAL / …)
3.  %include cloner / IO headers   ← functions that RETURN faiss::Index *
```

The GPU cloner (`GpuCloner.h`) already follows this convention.  The Metal
cloner (`MetalCloner.h`) was originally placed *before* the typemap, which
caused `index_cpu_to_metal` to return a generic `Index` proxy instead of the
proper `MetalIndexFlatL2` / `MetalIndexTurboQuantMSE` wrapper.  This was
fixed by moving the Metal `%newobject` and `%include <faiss/metal/MetalCloner.h>`
to after the `%typemap(out)` block.

### Quick check

If you add a new function that returns `faiss::Index *` and needs
downcasting, make sure its `%include` is placed **after** the
`%typemap(out) faiss::Index *` block (around line 830).
