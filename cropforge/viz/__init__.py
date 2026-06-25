"""
cropforge/viz/__init__.py
Visualization subpackage — Phase 2+3+4 Dashboard Frontend.
"""

from cropforge.viz.app import create_dash_app
from cropforge.viz.server import boot
from cropforge.viz.buffers import BUFFER_STORE, FIELD_REGISTRY

__all__ = ["create_dash_app", "boot", "BUFFER_STORE", "FIELD_REGISTRY"]
