import sys
import os
# Pre-import all six.moves-dependent libs before PySide6 to prevent shiboken conflict
try:
    import six as _six
    # Pre-load all six.moves attributes so they're cached before shiboken hooks
    import importlib
    for _attr in ['_thread', 'range', 'zip', 'map', 'filter', 'input',
                  'intern', 'reduce', 'reload_module', 'StringIO', 'BytesIO']:
        try:
            importlib.import_module('six.moves.' + _attr)
        except Exception:
            try:
                getattr(_six.moves, _attr)
            except Exception:
                pass
    import dateutil.tz
    import dateutil.rrule
    import dateutil.relativedelta
except ImportError:
    pass
import pandas  # noqa: F401
import matplotlib  # noqa: F401
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
try:
    import mplfinance  # noqa: F401
except Exception:
    pass
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from db.database import init_db, SessionLocal
from ui.main_window import MainWindow
from ui.styles.theme import get_stylesheet

def _install_exception_hook():
    import traceback
    _original = sys.excepthook
    def _hook(exc_type, exc_value, exc_tb):
        msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f"[UNHANDLED EXCEPTION]\n{msg}", flush=True)
        _original(exc_type, exc_value, exc_tb)
    sys.excepthook = _hook

    import threading
    _orig_thread_excepthook = getattr(threading, 'excepthook', None)
    def _thread_hook(args):
        msg = ''.join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        print(f"[THREAD EXCEPTION in {args.thread}]\n{msg}", flush=True)
        if _orig_thread_excepthook:
            _orig_thread_excepthook(args)
    threading.excepthook = _thread_hook

def main():
    _install_exception_hook()
    from app_paths import get_app_dir, get_config_path
    project_root = str(get_app_dir())
    
    app = QApplication(sys.argv)
    app.setApplicationName('SWS')
    app.setApplicationVersion('2.0.0')
    app.setOrganizationName('StockSense')
    
    # High DPI support
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    # Apply global stylesheet
    # Note: Using try-except because get_stylesheet() might rely on other files
    try:
        app.setStyleSheet(get_stylesheet())
    except Exception as e:
        print(f"Warning: Could not apply stylesheet: {e}")
    
    # Initialize database
    init_db()
    from db.models import AIScore
    from db.database import SessionLocal as SL
    _db = SL()
    _count = _db.query(AIScore).count()
    _db.close()
    if _count > 0:
        print(f'SWS: {_count} AI scores loaded.')
    else:
        print('SWS: No scores found. Running in demo mode with mock data.')

    db = SessionLocal()
    
    # Create and show main window
    window = MainWindow(db)
    window.show()
    
    # Wire up EOD Scheduler
    import json
    scheduler = None
    try:
        config_path = str(get_config_path())
        eod_enabled = True
        with open(config_path, 'r') as f:
            _cfg = json.load(f)
            eod_enabled = _cfg.get('eod_enabled', True)
        
        if eod_enabled:
            from scheduler.eod_scheduler import EODScheduler
            from services.alert_service import AlertService
            alert_svc = AlertService()
            scheduler = EODScheduler(db, alert_svc, config_path=config_path)
            scheduler.start()
    except ImportError:
        print("Warning: APScheduler not found. EOD Scheduler will not be started.")
    except Exception as e:
        print(f"Warning: Could not start EOD Scheduler: {e}")

    # Initial data load in background
    # This might take some time, so we call it after showing the window
    if hasattr(window.screening_page, 'refresh_data'):
        window.screening_page.refresh_data()
    
    ret = app.exec()

    if scheduler:
        try:
            scheduler.stop()
        except Exception:
            pass

    db.close()
    sys.exit(ret)

if __name__ == '__main__':
    main()
