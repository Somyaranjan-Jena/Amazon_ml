import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class Config:
    raw_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str = "business_entity_resolution/configs/default.yaml") -> "Config":
        if not os.path.exists(path):
            alt_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
            if os.path.exists(alt_path):
                path = alt_path
        with open(path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        return cls(raw_config=cfg_dict)

    def get(self, key_path: str, default: Any = None) -> Any:
        parts = key_path.split(".")
        val = self.raw_config
        for p in parts:
            if isinstance(val, dict) and p in val:
                val = val[p]
            else:
                return default
        return val

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw_config.get("data", {})

    @property
    def blocking(self) -> Dict[str, Any]:
        return self.raw_config.get("blocking", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw_config.get("model", {})

    @property
    def validation(self) -> Dict[str, Any]:
        return self.raw_config.get("validation", {})

    @property
    def decision(self) -> Dict[str, Any]:
        return self.raw_config.get("decision", {})
