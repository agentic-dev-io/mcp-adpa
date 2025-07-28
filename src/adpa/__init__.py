"""
A modern library for analyzing and indexing Python packages.

Provides AST-based analysis, DuckDB storage, and semantic search capabilities.
"""

__version__ = "0.1.0"

# Core functionality
from .models import FunctionInfo, ClassInfo, ModuleInfo, CallGraphInfo
from .analyzer import PackageASTAnalyzer
from .scanner import PackageScanner
from .indexer import PackageIndexer
from .context import CodeContextAnalyzer

# Production API
__all__ = [
    # Data models
    "FunctionInfo",
    "ClassInfo", 
    "ModuleInfo",
    "CallGraphInfo",
    
    # Core analysis
    "PackageASTAnalyzer",
    "PackageScanner",
    "PackageIndexer", 
    "CodeContextAnalyzer",
]

def get_version():
    """Get package version."""
    return __version__
