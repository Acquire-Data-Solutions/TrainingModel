# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

# 1. Gather all torch dependencies
datas, binaries, hiddenimports = [], [], []
torch_datas, torch_bins, torch_hidden = collect_all('torch')
datas     += torch_datas
binaries  += torch_bins
hiddenimports += torch_hidden

# 2. Analysis for trainModel.exe
a_train = Analysis(
    ['trainModel.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],
    noarchive=True,
    optimize=0,
)

# 3. Analysis for exportModel.exe
a_export = Analysis(
    ['exportModel.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],
    noarchive=True,
    optimize=0,
)

# 4. Create PYZ archives (one per entry point)
pyz_train  = PYZ(a_train.pure,  a_train.zipped_data,  cipher=None)
pyz_export = PYZ(a_export.pure, a_export.zipped_data, cipher=None)

# 5. Build EXEs (console=True ensures a visible window for LabVIEW RT)
exe_train = EXE(
    pyz_train,
    a_train.scripts,
    exclude_binaries=True,
    name='trainModel',
    debug=False,
    strip=False,
    upx=True,
    console=True,
)

exe_export = EXE(
    pyz_export,
    a_export.scripts,
    exclude_binaries=True,
    name='exportModel',
    debug=False,
    strip=False,
    upx=True,
    console=True,
)

# 6. Collect both executables into a single `_internal` folder
coll = COLLECT(
    exe_train,
    exe_export,
    a_train.binaries,
    a_train.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='.'
)