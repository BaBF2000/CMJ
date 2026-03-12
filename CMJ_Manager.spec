# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# ----------------------------
# Assets / project data files
# ----------------------------
datas = [
    # Root config files
    ("config/utils_config.json", "config"),
    ("config/utils_config.default.json", "config"),

    ("config/vicon_path_config.json", "config"),
    ("config/vicon_path_config.default.json", "config"),

    ("config/word_report_config.json", "config"),
    ("config/word_report_config.default.json", "config"),

    ("config/nexus_data_retrieval_config.json", "config"),
    ("config/nexus_data_retrieval_config.default.json", "config"),

    # GUI assets
    ("src/cmj_framework/gui/assets/icons/cmj_logo.ico", "src/cmj_framework/gui/assets/icons"),
    ("src/cmj_framework/gui/assets/styles/app.qss", "src/cmj_framework/gui/assets/styles"),

    # Export resources
    ("src/cmj_framework/export/resources/cmj_banner_placeholder.png", "resources"),
    ("src/cmj_framework/export/resources/parameter.md", "resources"),
    ("src/cmj_framework/export/resources/phases.md", "resources"),
    ("src/cmj_framework/export/resources/munster_graph.png", "resources"),

    # Vicon extraction script
    ("src/cmj_framework/vicon_data_retrieval/extraction.py", "src/cmj_framework/vicon_data_retrieval"),

    # Documentation
    ("documentation/CMJ_Framework_Documentation.html", "documentation"),
    ("documentation/images/documented_draw_window.png", "documentation/images"),
    ("documentation/images/documented_export_window.png", "documentation/images"),
    ("documentation/images/documented_folder_explorer.png", "documentation/images"),
    ("documentation/images/documented_parameter_window.png", "documentation/images"),
    ("documentation/images/documented_welcome_window.png", "documentation/images"),
]

binaries = []

# ----------------------------
# Excludes (shrink + stability)
# ----------------------------
excludes = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.scripts",
    "PySide6.scripts.deploy_lib",
]

# ----------------------------
# Hidden imports
# ----------------------------
hiddenimports = []
hiddenimports += collect_submodules("PySide6.QtCore")
hiddenimports += collect_submodules("PySide6.QtGui")
hiddenimports += collect_submodules("PySide6.QtWidgets")
hiddenimports += collect_submodules("PySide6.QtSvg")
hiddenimports += collect_submodules("PySide6.QtPrintSupport")
hiddenimports += collect_submodules("matplotlib.backends")
hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("scipy")

datas += collect_data_files("PySide6", include_py_files=False)

a = Analysis(
    ["src/cmj_framework/gui/app.pyw"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CMJ_Manager",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=["src/cmj_framework/gui/assets/icons/cmj_logo.ico"],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="CMJ_Manager",
)