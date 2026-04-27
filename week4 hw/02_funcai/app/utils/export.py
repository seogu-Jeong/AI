"""ExportManager — PNG and JSON export."""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from PySide6.QtWidgets import QFileDialog


class ExportManager:
    @staticmethod
    def export_png(mpl_widget, module_id: str, parent=None) -> Optional[str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            parent, "Export PNG",
            str(Path.home() / "Downloads" / f"{module_id}_{ts}.png"),
            "PNG (*.png)")
        if path: mpl_widget.figure.savefig(path, dpi=150, bbox_inches='tight')
        return path or None

    @staticmethod
    def export_json(module, parent=None) -> Optional[str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            parent, "Export JSON",
            str(Path.home() / "Downloads" / f"{getattr(module,'MODULE_ID','MOD')}_{ts}.json"),
            "JSON (*.json)")
        if not path: return None
        data = {
            "schema_version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "module": {"id": getattr(module,'MODULE_ID',''), "name": getattr(module,'MODULE_NAME','')},
            "parameters": module.get_param_values() if hasattr(module,'get_param_values') else {},
            "metrics": module.get_metrics() if hasattr(module,'get_metrics') else {},
        }
        with open(path, 'w') as f: json.dump(data, f, indent=2, default=str)
        return path
