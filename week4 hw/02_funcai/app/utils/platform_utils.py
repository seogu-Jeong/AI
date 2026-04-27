"""Korean font detection for Matplotlib."""
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def configure_korean_font():
    PRIORITY = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'Gulim', 'DejaVu Sans']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in PRIORITY:
        if font in available:
            plt.rcParams['font.family'] = font; break
    plt.rcParams['axes.unicode_minus'] = False
