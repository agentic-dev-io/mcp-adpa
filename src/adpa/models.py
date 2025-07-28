from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CallGraphInfo:
    """Information about a function call."""
    source_function: str
    target_function: str
    module: str
    package: str

@dataclass
class FunctionInfo:
    """Information about a function - extended for Python 3.13+"""
    name: str
    module: str
    package: str
    docstring: Optional[str]
    signature: str
    parameters: List[str]
    return_annotation: Optional[str]
    source_file: Optional[str]
    line_number: int
    end_line_number: Optional[int]  # Python 3.8+
    column_offset: int
    end_column_offset: Optional[int]  # Python 3.8+
    complexity_score: int
    ast_hash: str
    decorators: List[str]
    is_async: bool
    type_comments: Optional[str]  # Python 3.8+
    
@dataclass
class ClassInfo:
    """Information about a class - extended"""
    name: str
    module: str
    package: str
    docstring: Optional[str]
    methods: List[str]
    base_classes: List[str]
    source_file: Optional[str]
    line_number: int
    end_line_number: Optional[int]
    decorators: List[str]
    ast_hash: str
    is_dataclass: bool
    type_params: List[str]  # Python 3.12+ Type Parameters

@dataclass
class ModuleInfo:
    """Information about a module - extended"""
    name: str
    package: str
    docstring: Optional[str]
    imports: List[str]
    functions: List[str]
    classes: List[str]
    source_file: Optional[str]
    ast_hash: str
    encoding: str
    future_imports: List[str]  # __future__ imports
    type_aliases: List[str]  # Python 3.12+ Type Aliases
