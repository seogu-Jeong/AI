"""
Platform utilities — font detection, OS-specific helpers.
All platform-specific code is isolated here (TRD-01 §5.4).
"""
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def configure_korean_font():
    """Detects and sets the best available Korean-compatible font for Matplotlib."""
    PRIORITY = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'Gulim', 'DejaVu Sans']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in PRIORITY:
        if font in available:
            plt.rcParams['font.family'] = font
            break
    plt.rcParams['axes.unicode_minus'] = False
