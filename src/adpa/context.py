import ast
from typing import Any, Dict, List

from .indexer import PackageIndexer


class CodeContextAnalyzer:
    """Context analysis with current features"""
    
    def __init__(self, indexer: PackageIndexer):
        self.indexer = indexer
    
    def analyze_patterns(self, source_code: str) -> Dict[str, Any]:
        """Analyzes Python patterns"""
        
        tree = ast.parse(source_code)
        patterns = {
            'async_usage': self._analyze_async_patterns(tree),
            'type_hints': self._analyze_type_hints(tree),
            'decorators': self._analyze_decorators(tree),
            'comprehensions': self._analyze_comprehensions(tree),
            'context_managers': self._analyze_context_managers(tree),
            'future_imports': self._analyze_future_imports(tree)
        }
        
        return patterns
    
    def _analyze_async_patterns(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyzes Async/Await patterns"""
        async_functions = 0
        await_calls = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                async_functions += 1
            elif isinstance(node, ast.Await):
                await_calls += 1
        
        return {
            'async_functions': async_functions,
            'await_calls': await_calls,
            'async_ratio': await_calls / max(async_functions, 1)
        }
    
    def _analyze_type_hints(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyzes Type Hints usage"""
        annotated_functions = 0
        total_functions = 0
        annotated_variables = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_functions += 1
                if node.returns or any(arg.annotation for arg in node.args.args):
                    annotated_functions += 1
            elif isinstance(node, ast.AnnAssign):
                annotated_variables += 1
        
        return {
            'annotated_functions': annotated_functions,
            'total_functions': total_functions,
            'annotation_coverage': annotated_functions / max(total_functions, 1),
            'annotated_variables': annotated_variables
        }
    
    def _analyze_decorators(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyzes Decorator usage"""
        decorators = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    decorators.append(ast.unparse(decorator))
        
        decorator_counts: Dict[str, int] = {}
        for dec in decorators:
            decorator_counts[dec] = decorator_counts.get(dec, 0) + 1
        
        return {
            'total_decorators': len(decorators),
            'unique_decorators': len(decorator_counts),
            'decorator_distribution': decorator_counts
        }
    
    def _analyze_comprehensions(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyzes Comprehension usage"""
        comprehension_types = {
            'list': 0, 'dict': 0, 'set': 0, 'generator': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ListComp):
                comprehension_types['list'] += 1
            elif isinstance(node, ast.DictComp):
                comprehension_types['dict'] += 1
            elif isinstance(node, ast.SetComp):
                comprehension_types['set'] += 1
            elif isinstance(node, ast.GeneratorExp):
                comprehension_types['generator'] += 1
        
        return comprehension_types
    
    def _analyze_context_managers(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyzes Context Manager usage"""
        with_statements = 0
        async_with_statements = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                with_statements += 1
            elif isinstance(node, ast.AsyncWith):
                async_with_statements += 1
        
        return {
            'with_statements': with_statements,
            'async_with_statements': async_with_statements,
            'total_context_managers': with_statements + async_with_statements
        }
    
    def _analyze_future_imports(self, tree: ast.AST) -> List[str]:
        """Analyzes __future__ imports"""
        future_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == '__future__':
                for alias in node.names:
                    future_imports.append(alias.name)
        
        return future_imports
