"""
Preset management for Null-Hound.

Presets define language/framework-specific configurations for file filtering
and analysis.
"""

import json
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


class PresetLoader:
    """Load and manage filter presets."""

    def __init__(self):
        self.presets_dir = Path(__file__).parent.parent / "presets"
        self._cache: dict[str, dict[str, Any]] = {}

    def list_presets(self) -> list[str]:
        """List available preset names."""
        if not self.presets_dir.exists():
            return []
        return [p.stem for p in self.presets_dir.glob("*.json")]

    def load(self, name: str) -> dict[str, Any]:
        """Load a preset by name.

        Args:
            name: Preset name (without .json extension)

        Returns:
            Preset configuration dictionary

        Raises:
            FileNotFoundError: If preset doesn't exist
            ValueError: If preset JSON is invalid
        """
        # Check cache
        if name in self._cache:
            return self._cache[name]

        preset_path = self.presets_dir / f"{name}.json"
        if not preset_path.exists():
            available = self.list_presets()
            raise FileNotFoundError(
                f"Preset '{name}' not found. Available presets: {', '.join(available)}"
            )

        try:
            with open(preset_path) as f:
                preset = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in preset '{name}': {e}")

        # Validate required fields
        required = ["name", "extensions", "path_boosts", "path_penalties"]
        missing = [f for f in required if f not in preset]
        if missing:
            raise ValueError(f"Preset '{name}' missing required fields: {', '.join(missing)}")

        # Cache and return
        self._cache[name] = preset
        return preset

    def get_extension_weights(self, preset: dict[str, Any]) -> dict[str, float]:
        """Get extension weights from preset."""
        return {k: float(v) for k, v in preset.get("extensions", {}).items()}

    def get_path_boosts(self, preset: dict[str, Any]) -> list[tuple[str, float]]:
        """Get path boost patterns and weights."""
        return [
            (item["pattern"], float(item["weight"]))
            for item in preset.get("path_boosts", [])
        ]

    def get_path_penalties(self, preset: dict[str, Any]) -> list[tuple[str, float]]:
        """Get path penalty patterns and weights."""
        return [
            (item["pattern"], float(item["weight"]))
            for item in preset.get("path_penalties", [])
        ]

    def get_config_top_files(self, preset: dict[str, Any]) -> set[str]:
        """Get top-level config file names."""
        return set(preset.get("config_top_files", []))

    def get_entrypoint_patterns(self, preset: dict[str, Any]) -> list[str]:
        """Get entrypoint regex patterns."""
        return preset.get("entrypoint_patterns", [])

    def get_llm_prompts(self, preset: dict[str, Any]) -> tuple[str, str]:
        """Get LLM system prompt and instructions.

        Returns:
            (system_prompt, instructions) tuple
        """
        system = preset.get(
            "llm_system_prompt",
            "You prioritize source files for a security audit whitelist."
        )
        instructions = preset.get(
            "llm_instructions",
            'Return JSON with a single field "prioritized", an array of relpaths, in priority order.'
        )
        return system, instructions


def get_preset_loader() -> PresetLoader:
    """Get singleton preset loader instance."""
    if not hasattr(get_preset_loader, "_instance"):
        get_preset_loader._instance = PresetLoader()
    return get_preset_loader._instance