"""
ExportManager — PNG and JSON export pipeline.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QFileDialog


class ExportManager:

    @staticmethod
    def export_png(mpl_widget, module_id: str, parent=None) -> Optional[str]:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{module_id}_{ts}.png"
        dest = str(Path.home() / "Downloads")
        path, _ = QFileDialog.getSaveFileName(
            parent, "Export Plot as PNG",
            str(Path(dest) / name),
            "PNG Image (*.png);;All Files (*)",
        )
        if path:
            mpl_widget.figure.savefig(path, dpi=150, bbox_inches='tight')
        return path or None

    @staticmethod
    def export_json(module, parent=None) -> Optional[str]:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{module.MODULE_ID}_{ts}.json"
        dest = str(Path.home() / "Downloads")
        path, _ = QFileDialog.getSaveFileName(
            parent, "Export Results as JSON",
            str(Path(dest) / name),
            "JSON File (*.json);;All Files (*)",
        )
        if not path:
            return None
        data = {
            "schema_version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "module": {
                "id":   getattr(module, 'MODULE_ID',   'UNKNOWN'),
                "name": getattr(module, 'MODULE_NAME', 'Unknown'),
            },
            "parameters": module.get_param_values() if hasattr(module, 'get_param_values') else {},
            "training": {
                "epochs_run":      getattr(module, 'epochs_run', None),
                "final_loss":      getattr(module, 'final_loss', None),
                "final_val_loss":  getattr(module, 'final_val_loss', None),
                "elapsed_ms":      getattr(module, 'training_elapsed_ms', None),
            },
            "metrics": module.get_metrics() if hasattr(module, 'get_metrics') else {},
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return path
