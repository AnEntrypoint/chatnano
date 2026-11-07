import importlib.metadata
import os
import sysconfig

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

REPO = os.path.dirname(SPECPATH)

metadata = []
for name in sorted({d.metadata["Name"] for d in importlib.metadata.distributions() if d.metadata["Name"]}):
    try:
        metadata += copy_metadata(name)
    except Exception:
        pass

datas, binaries, hiddenimports = list(metadata), [], []
datas.append((sysconfig.get_paths()["include"], os.path.join("include", "python" + sysconfig.get_python_version())))

triton_datas, triton_binaries, triton_hidden = collect_all("triton")
datas += triton_datas
binaries += triton_binaries
hiddenimports += triton_hidden

for package in ("torch._inductor", "transformers", "librosa", "s3tokenizer", "perth", "misaki"):
    hiddenimports += collect_submodules(package)
    datas += collect_data_files(package)

EXCLUDES = [
    "tkinter", "matplotlib", "gradio", "gradio_client", "tensorboard", "lightning",
    "pytorch_lightning", "IPython", "pytest", "onnxruntime",
]

a = Analysis(
    [os.path.join(SPECPATH, "serve.py")],
    pathex=[os.path.join(REPO, "chatterbox", "src"), os.path.join(REPO, "nano_fast")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="chatnano-serve",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="chatnano-serve")
