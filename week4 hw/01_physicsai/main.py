"""
PhysicsAI Simulator — entry point.
matplotlib backend MUST be set before any Qt import (TRD-01 §5.3 / DEP-02).
matplotlib must also be fully initialized before PySide6 to avoid six.moves
shiboken interception bug.
"""
import matplotlib
matplotlib.use('QtAgg')
# Force complete matplotlib import chain (including dateutil) BEFORE PySide6
# to avoid shiboken's six.moves interception (PySide6 >= 6.5 known issue).
import matplotlib.pyplot  # noqa: F401 — side-effect import

# Pre-import TensorFlow before PySide6 for same reason
try:
    import tensorflow as _tf  # noqa: F401
    _tf.get_logger().setLevel('ERROR')
    del _tf
except Exception:
    pass  # TF not available; handled later in _configure_tensorflow()

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
)

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from app.main_window import MainWindow
from app.utils.theme import ThemeManager
from app.utils.platform_utils import configure_korean_font


def _configure_tensorflow():
    """GPU memory growth setup. Non-blocking — app continues on failure."""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            logging.info(f"TensorFlow GPU(s): {[g.name for g in gpus]}")
        else:
            logging.info("TensorFlow: no GPU found, running on CPU")
    except ImportError:
        logging.warning("TensorFlow not installed — training will not work")
    except Exception as exc:
        logging.warning(f"TF GPU config failed: {exc}. CPU mode active.")


def main():
    # ── High-DPI / platform attributes ────────────────────────────────────────
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("PhysicsAI Simulator")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("JSW Lab")

    # Default font — use platform system font
    font = app.font()
    font.setPointSize(11)
    app.setFont(font)

    # ── Setup ──────────────────────────────────────────────────────────────────
    configure_korean_font()
    ThemeManager.apply_light(app)
    _configure_tensorflow()

    # ── Main window ────────────────────────────────────────────────────────────
    window = MainWindow()
    window.show()
    window.raise_()          # bring to front
    window.activateWindow()  # focus (macOS: also needs Dock activation)

    # macOS: activate via NSApp so the window appears in front of other apps
    try:
        from AppKit import NSApplication  # type: ignore
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
    except ImportError:
        pass  # not on macOS or pyobjc not installed — window still opens

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
