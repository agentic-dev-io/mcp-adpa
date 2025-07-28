import ast
import hashlib
from typing import Dict, List, Optional, Set, Any
from .models import FunctionInfo, ClassInfo, CallGraphInfo

class PackageASTAnalyzer(ast.NodeVisitor):
    """Modern AST Analyzer using NodeVisitor pattern with best practices for 2025."""

    def __init__(self, module_name: str, package_name: str, source_file: Optional[str] = None):
        super().__init__()
        self.module_name = module_name
        self.package_name = package_name
        self.source_file = source_file

        # Analysis results storage
        self.functions: List[FunctionInfo] = []
        self.classes: List[ClassInfo] = []
        self.imports: List[str] = []
        self.future_imports: List[str] = []
        self.type_aliases: List[str] = []
        self.call_graph: List[CallGraphInfo] = []
        
        # State tracking for visitor pattern
        self._current_function: Optional[FunctionInfo] = None
        self._current_class: Optional[str] = None
        self._scope_stack: List[str] = []
        
        # Performance optimizations
        self._visited_nodes: Set[int] = set()
        self._cached_signatures: Dict[int, str] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process regular function definitions."""
        self._process_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Process async function definitions."""
        self._process_function(node, is_async=True)

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> None:
        """Process function nodes with comprehensive analysis and caching."""
        node_id = id(node)
        
        # Avoid reprocessing the same node
        if node_id in self._visited_nodes:
            return
        self._visited_nodes.add(node_id)
        
        # Build function signature with caching
        signature = self._get_or_create_signature(node, node_id, is_async)
        
        # Extract comprehensive node information
        func_info = FunctionInfo(
            name=node.name,
            module=self.module_name,
            package=self.package_name,
            docstring=ast.get_docstring(node),
            signature=signature,
            parameters=self._extract_parameters(node),
            return_annotation=self._safe_unparse(node.returns),
            source_file=self.source_file,
            line_number=getattr(node, 'lineno', 0),
            end_line_number=getattr(node, 'end_lineno', None),
            column_offset=getattr(node, 'col_offset', 0),
            end_column_offset=getattr(node, 'end_col_offset', None),
            complexity_score=self._calculate_complexity(node),
            ast_hash=self._generate_ast_hash(node),
            decorators=self._extract_decorators(node),
            is_async=is_async,
            type_comments=getattr(node, 'type_comment', None)
        )

        self.functions.append(func_info)
        
        # Manage function scope for call graph analysis
        self._enter_function_scope(func_info)
        self.generic_visit(node)
        self._exit_function_scope()
        
    def _get_or_create_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef, 
                                node_id: int, is_async: bool) -> str:
        """Get cached signature or create new one."""
        if node_id in self._cached_signatures:
            return self._cached_signatures[node_id]
            
        args = self._build_argument_list(node.args)
        signature = f"{node.name}({', '.join(args)})"
        
        if node.returns:
            signature += f" -> {self._safe_unparse(node.returns)}"
            
        if is_async:
            signature = f"async {signature}"
            
        self._cached_signatures[node_id] = signature
        return signature
        
    def _build_argument_list(self, args: ast.arguments) -> List[str]:
        """Build formatted argument list with type annotations."""
        arg_strings = []
        
        # Regular arguments
        for arg in args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._safe_unparse(arg.annotation)}"
            arg_strings.append(arg_str)
            
        # *args
        if args.vararg:
            vararg_str = f"*{args.vararg.arg}"
            if args.vararg.annotation:
                vararg_str += f": {self._safe_unparse(args.vararg.annotation)}"
            arg_strings.append(vararg_str)
            
        # **kwargs  
        if args.kwarg:
            kwarg_str = f"**{args.kwarg.arg}"
            if args.kwarg.annotation:
                kwarg_str += f": {self._safe_unparse(args.kwarg.annotation)}"
            arg_strings.append(kwarg_str)
            
        return arg_strings
        
    def _extract_parameters(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
        """Extract parameter names."""
        params = [arg.arg for arg in node.args.args]
        if node.args.vararg:
            params.append(node.args.vararg.arg)
        if node.args.kwarg:
            params.append(node.args.kwarg.arg)
        return params
        
    def _extract_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> List[str]:
        """Extract decorator information safely."""
        return [self._safe_unparse(decorator) for decorator in node.decorator_list]
        
    def _enter_function_scope(self, func_info: FunctionInfo) -> None:
        """Enter function scope for call graph tracking."""
        self._current_function = func_info
        self._scope_stack.append(func_info.name)
        
    def _exit_function_scope(self) -> None:
        """Exit function scope."""
        self._current_function = None
        if self._scope_stack:
            self._scope_stack.pop()


    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Process class definitions with comprehensive analysis."""
        node_id = id(node)
        
        if node_id in self._visited_nodes:
            return
        self._visited_nodes.add(node_id)
        
        class_info = ClassInfo(
            name=node.name,
            module=self.module_name,
            package=self.package_name,
            docstring=ast.get_docstring(node),
            methods=self._extract_methods(node),
            base_classes=self._extract_base_classes(node),
            source_file=self.source_file,
            line_number=getattr(node, 'lineno', 0),
            end_line_number=getattr(node, 'end_lineno', None),
            decorators=self._extract_decorators(node),
            ast_hash=self._generate_ast_hash(node),
            is_dataclass=self._is_dataclass(node),
            type_params=self._extract_type_params(node)
        )

        self.classes.append(class_info)
        
        # Enter class scope
        previous_class = self._current_class
        self._current_class = node.name
        self._scope_stack.append(node.name)
        
        self.generic_visit(node)
        
        # Exit class scope
        self._current_class = previous_class
        if self._scope_stack:
            self._scope_stack.pop()
            
    def _extract_methods(self, node: ast.ClassDef) -> List[str]:
        """Extract method names from class definition."""
        return [
            item.name for item in node.body 
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        
    def _extract_base_classes(self, node: ast.ClassDef) -> List[str]:
        """Extract base class names."""
        return [self._safe_unparse(base) for base in node.bases]
        
    def _is_dataclass(self, node: ast.ClassDef) -> bool:
        """Check if class is decorated with @dataclass."""
        decorators = self._extract_decorators(node)
        return any('dataclass' in dec.lower() for dec in decorators)
        
    def _extract_type_params(self, node: ast.ClassDef) -> List[str]:
        """Extract type parameters for Python 3.12+."""
        if not hasattr(node, 'type_params'):
            return []
        return [self._safe_unparse(param) for param in node.type_params]

    def visit_Import(self, node: ast.Import) -> None:
        """Process import statements."""
        for alias in node.names:
            import_name = alias.asname if alias.asname else alias.name
            self.imports.append(import_name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Process from-import statements with special handling."""
        module = node.module or ""

        if module == '__future__':
            # Track __future__ imports separately
            self.future_imports.extend(alias.name for alias in node.names)
        else:
            # Regular imports with qualified names
            for alias in node.names:
                import_name = alias.asname if alias.asname else alias.name
                if module:
                    self.imports.append(f"{module}.{import_name}")
                else:
                    self.imports.append(import_name)
        self.generic_visit(node)

    def visit_TypeAlias(self, node) -> None:
        """Process Type Aliases (Python 3.12+ feature)."""
        if hasattr(ast, 'TypeAlias') and isinstance(node, ast.TypeAlias):
            alias_name = self._safe_unparse(node.name)
            self.type_aliases.append(alias_name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Process function calls for call graph analysis."""
        if self._current_function:
            target_name = self._resolve_call_target(node.func)
            if target_name:
                call_info = CallGraphInfo(
                    source_function=self._current_function.name,
                    target_function=target_name,
                    module=self._current_function.module,
                    package=self._current_function.package,
                )
                self.call_graph.append(call_info)
        self.generic_visit(node)
        
    def _resolve_call_target(self, func_node: ast.expr) -> Optional[str]:
        """Resolve the target function name from a call expression."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            # Method calls: obj.method() -> "method"
            return func_node.attr
        elif isinstance(func_node, ast.Subscript):
            # Generic calls: Callable[...]() -> try to resolve
            return self._resolve_call_target(func_node.value)
        return None

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity with comprehensive metrics."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)  # Each except clause
                if child.orelse:
                    complexity += 1  # else clause
                if child.finalbody:
                    complexity += 1  # finally clause
            elif isinstance(child, ast.With):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1  # AND/OR operations
            elif isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1
            elif isinstance(child, ast.Lambda):
                complexity += 1
            elif isinstance(child, (ast.Break, ast.Continue)):
                complexity += 1
                
        return complexity
        
    def _safe_unparse(self, node: Optional[ast.AST]) -> Optional[str]:
        """Safely unparse AST node with error handling."""
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except Exception:
            # Fallback for unparseable nodes
            return str(type(node).__name__)
            
    def _generate_ast_hash(self, node: ast.AST) -> str:
        """Generate stable hash for AST node."""
        try:
            ast_dump = ast.dump(node, indent=None)
            return hashlib.sha256(ast_dump.encode('utf-8')).hexdigest()[:16]
        except Exception:
            # Fallback to id-based hash
            return hashlib.sha256(str(id(node)).encode()).hexdigest()[:16]
            
    def analyze(self, source_code: str) -> Dict[str, Any]:
        """Main entry point for analysis."""
        try:
            tree = ast.parse(source_code)
            self.visit(tree)
            
            return {
                'functions': self.functions,
                'classes': self.classes,
                'imports': self.imports,
                'future_imports': self.future_imports,
                'type_aliases': self.type_aliases,
                'call_graph': self.call_graph,
                'stats': {
                    'total_functions': len(self.functions),
                    'total_classes': len(self.classes),
                    'total_imports': len(self.imports),
                    'avg_complexity': sum(f.complexity_score for f in self.functions) / len(self.functions) if self.functions else 0
                }
            }
        except SyntaxError as e:
            raise ValueError(f"Syntax error in source code: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {e}") from e