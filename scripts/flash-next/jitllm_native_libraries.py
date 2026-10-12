"""Private unversioned CUDA aliases for the pinned SDK's FFM library bindings."""
from pathlib import Path


def prepare_native_aliases(native: Path, directory: Path) -> Path:
    native = native.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    origins = {"libcudnn.so": native / "cudnn/lib/libcudnn.so.9",
               "libcudart.so": native / "cu13/lib/libcudart.so.13",
               "libnvrtc.so": native / "cu13/lib/libnvrtc.so.13",
               "libcublas.so": native / "cu13/lib/libcublas.so.13",
               "libcublasLt.so": native / "cu13/lib/libcublasLt.so.13"}
    # Preflight every path before making changes; occupied paths belong to the caller.
    for name, origin in origins.items():
        link = directory / name
        if not origin.is_file():
            raise RuntimeError(f"Missing native CUDA dependency: {origin}")
        if link.is_symlink():
            if link.resolve() != origin.resolve():
                raise RuntimeError(f"Native alias points elsewhere: {link}")
        elif link.exists():
            raise RuntimeError(f"Native alias path is occupied: {link}")
    for name, origin in origins.items():
        link = directory / name
        if not link.is_symlink():
            link.symlink_to(origin)
    return directory
