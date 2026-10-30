# -*- coding: utf-8 -*-
"""CJK 폰트 등록"""
import os
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_FONT_PATHS = {
    "kr": os.environ.get("ESG_FONT_KR", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
}
if os.environ.get("ESG_RESEARCH_ROOT"):
    _FONT_PATHS["jp"] = str(
        Path(os.environ["ESG_RESEARCH_ROOT"]).expanduser()
        / "japan" / "preprocessing" / "fonts" / "HiraginoSans.ttf"
    )
if os.environ.get("ESG_FONT_JP"):
    _FONT_PATHS["jp"] = os.environ["ESG_FONT_JP"]
_FONT_NAMES = {}
for _lang, _path in _FONT_PATHS.items():
    _path = str(Path(_path).expanduser())
    _FONT_PATHS[_lang] = _path
    if Path(_path).exists():
        fm.fontManager.addfont(_path)
        _FONT_NAMES[_lang] = fm.FontProperties(fname=_path).get_name()

plt.rcParams["axes.unicode_minus"] = False


def set_font(lang: str):
    """figure 폰트를 언어별로 설정."""
    name = _FONT_NAMES.get(lang)
    if name:
        plt.rcParams["font.family"] = name
