import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.config import EXPERIMENTS_RESULTS_JSON, EXPERIMENTS_DIR


class ExperimentTracker:
    """
    Records, updates, and retrieves experiment configurations and evaluation results.
    """

    def __init__(self, filepath: Path = EXPERIMENTS_RESULTS_JSON):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.experiments = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not read {self.filepath}: {e}")
                return []
        return []

    def save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.experiments, f, indent=2)

    def log_experiment(
        self,
        experiment_id: str,
        name: str,
        approach_type: str,
        model_name: str,
        hyperparameters: Dict[str, Any],
        val_metrics: Optional[Dict[str, float]] = None,
        test_metrics: Optional[Dict[str, float]] = None,
        notes: str = "",
        validation_metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Logs a single experiment result. If experiment_id exists, updates it.
        """
        effective_val = val_metrics if val_metrics is not None else (validation_metrics or {})
        record = {
            "experiment_id": experiment_id,
            "name": name,
            "approach_type": approach_type,  # e.g. "Baseline", "Text DL", "Image DL", "Multimodal", "Optimized"
            "model_name": model_name,
            "hyperparameters": hyperparameters,
            "validation_metrics": effective_val,
            "test_metrics": test_metrics or {},
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
        }

        # Check for existing record
        for idx, item in enumerate(self.experiments):
            if item.get("experiment_id") == experiment_id:
                self.experiments[idx] = record
                self.save()
                return record

        self.experiments.append(record)
        self.save()
        return record

    def get_all(self) -> List[Dict[str, Any]]:
        return self.experiments

    def get_by_id(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        for item in self.experiments:
            if item.get("experiment_id") == experiment_id:
                return item
        return None
