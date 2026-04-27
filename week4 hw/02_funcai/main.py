"""
FuncAI Studio — entry point.
matplotlib backend set before Qt; TF imported before PySide6 (six.moves workaround).
"""
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot   # force full import before PySide6 (shiboken/six fix)

try:
    import tensorflow as _tf; _tf.get_logger().setLevel('ERROR'); del _tf
except Exception: pass

import sys, logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from app.main_window import MainWindow
from app.utils.theme import ThemeManager
from app.utils.platform_utils import configure_korean_font


def _configure_tensorflow():
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        gpus = tf.config.list_physical_devices('GPU')
        logging.info(f"TF GPU(s): {[g.name for g in gpus]}" if gpus else "TF: CPU mode")
    except Exception as exc:
        logging.warning(f"TF config: {exc}")


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("FuncAI Studio")
    app.setApplicationVersion("1.0.0")
    font = app.font(); font.setPointSize(11); app.setFont(font)

    configure_korean_font()
    ThemeManager.apply_light(app)
    _configure_tensorflow()

    window = MainWindow()
    window.show(); window.raise_(); window.activateWindow()
    try:
        from AppKit import NSApplication
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
    except ImportError: pass

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
