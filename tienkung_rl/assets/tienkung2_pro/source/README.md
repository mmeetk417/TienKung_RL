# TienKung 2 Pro source asset map

This directory is the immutable input set for the MuJoCo asset generator.
The URDF and every STL are byte-for-byte copies of the archived pre-migration
asset at:

```text
legacy/isaaclab/source/legged_lab/assets/tienkung2_pro/
```

The URDF remains the source of truth for link/joint hierarchy, axes, limits,
inertial data, visual geometry references, and hand `mimic` relationships.
Its historical `../meshes/*.STL` references are resolved by the generator to
this directory's `meshes/` inventory without rewriting the source file.  This
mapping is recorded in `source_manifest.json`.

`LICENSE` preserves the repository's BSD-3-Clause notices, including the
original RSL-RL, Isaac Lab, Legged Lab, and TienKung-Lab attributions.

Do not edit these inputs in place.  Replace them from an attributable upstream
source and regenerate all manifests and MJCF outputs instead.
