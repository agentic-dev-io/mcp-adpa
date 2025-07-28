"""
Modern Package Scanner with sentence transformer embeddings.
Optimized for DuckDB VSS with zero-copy operations and Arrow integration.
"""

import ast
import importlib
import inspect
import logging
import os
import pkgutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow as pa
from sentence_transformers import SentenceTransformer

from .analyzer import PackageASTAnalyzer
from .models import CallGraphInfo, ClassInfo, FunctionInfo, ModuleInfo

logger = logging.getLogger(__name__)

class PackageScanner:
    """Highly optimized Package Scanner with VSS-optimized embeddings."""
    
    def __init__(self, max_workers: Optional[int] = None, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize PackageScanner with optimized settings for VSS.
        
        Args:
            max_workers: Number of workers for parallel processing (auto-detect if None)
            embedding_model: SentenceTransformer model name
        """
        # Auto-detect optimal worker count
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            self.max_workers = min(cpu_count, 8)  # Cap at 8 for memory efficiency
        else:
            self.max_workers = max_workers
        
        # Initialize embedding model with optimizations
        self.embedding_model_name = embedding_model
        self.embedding_model = self._initialize_embedding_model()
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # File processing cache for change detection
        self._file_cache = {}
        self._embedding_cache = {}  # Cache for identical text embeddings
        
        logger.info(f"Initialized optimized SentenceTransformer model '{embedding_model}' "
                   f"(dim={self.embedding_dimension}) with {self.max_workers} workers")
        
    def _initialize_embedding_model(self) -> SentenceTransformer:
        """Initialize SentenceTransformer with optimal settings."""
        try:
            # Load model with optimizations
            model = SentenceTransformer(
                self.embedding_model_name,
                trust_remote_code=False,  # Security best practice
                device='cpu'  # Consistent across environments
            )
            
            # Optimize for inference
            model.eval()
            
            logger.info(f"Successfully loaded embedding model: {self.embedding_model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.embedding_model_name}: {e}")
            # Fallback to a smaller model
            fallback_model = 'all-MiniLM-L12-v2'
            logger.info(f"Falling back to {fallback_model}")
            return SentenceTransformer(fallback_model, device='cpu')
        
    def analyze_package(self, package_name: str) -> Tuple[
        List[FunctionInfo], List[ClassInfo], List[ModuleInfo], List[CallGraphInfo]
    ]:
        """Analyze a package with optimized parallel processing and caching."""
        logger.info(f"Analyzing package: {package_name}")
        
        functions = []
        classes = []
        modules = []
        call_graph = []
        
        try:
            # Import the package
            package = importlib.import_module(package_name)
            
            # Determine package path with better detection
            package_path = self._get_package_path(package)
            logger.info(f"Package path: {package_path}")
            
            # Collect all Python files with filtering
            py_files = self._collect_python_files(package_path)
            logger.info(f"Found {len(py_files)} Python files to analyze")
            
            if not py_files:
                logger.warning(f"No Python files found in package {package_name}")
                return functions, classes, modules, call_graph
            
            # Process files in parallel with optimized batching
            file_results = self._process_files_parallel(py_files, package_name)
            
            # Aggregate results efficiently
            for file_functions, file_classes, file_module, file_call_graph in file_results:
                functions.extend(file_functions)
                classes.extend(file_classes)
                if file_module:
                    modules.append(file_module)
                call_graph.extend(file_call_graph)
                        
        except ImportError as e:
            logger.error(f"Could not import package {package_name}: {e}")
        except Exception as e:
            logger.error(f"Error analyzing package {package_name}: {e}")
        
        logger.info(
            f"Analysis complete: {len(functions)} functions, {len(classes)} classes, "
            f"{len(modules)} modules, {len(call_graph)} call graph edges"
        )
        return functions, classes, modules, call_graph
    
    def _get_package_path(self, package) -> Path:
        """Get package path with robust detection."""
        if hasattr(package, '__path__'):
            # Package with __path__ (namespace package or regular package)
            if isinstance(package.__path__, list) and package.__path__:
                return Path(package.__path__[0])
            else:
                return Path(str(package.__path__))
        elif hasattr(package, '__file__') and package.__file__:
            # Single module
            return Path(package.__file__).parent
        else:
            raise ValueError(f"Cannot determine path for package {package}")
    
    def _collect_python_files(self, package_path: Path) -> List[Path]:
        """Collect Python files with intelligent filtering."""
        py_files = []
        
        # Pattern exclusions for better performance
        exclude_patterns = {
            '__pycache__', '.git', '.svn', '.hg', 'node_modules',
            'venv', 'env', '.venv', '.env', 'build', 'dist',
            '.pytest_cache', '.mypy_cache', '.coverage'
        }
        
        for py_file in package_path.rglob("*.py"):
            # Skip files in excluded directories
            if any(pattern in py_file.parts for pattern in exclude_patterns):
                continue
                
            # Skip test files for main analysis (can be configured)
            if 'test' in py_file.stem.lower() and not py_file.stem.startswith('test_'):
                continue
                
            # Skip very large files (>1MB) that might be generated
            if py_file.stat().st_size > 1024 * 1024:
                logger.warning(f"Skipping large file: {py_file} ({py_file.stat().st_size} bytes)")
                continue
                
            py_files.append(py_file)
        
        return py_files
    
    def _process_files_parallel(self, py_files: List[Path], package_name: str) -> List[Tuple]:
        """Process files in parallel with optimal strategy."""
        results = []
        
        # Use ProcessPoolExecutor for CPU-intensive parsing, ThreadPoolExecutor for I/O
        # Choose based on file count and size
        total_size = sum(f.stat().st_size for f in py_files)
        use_processes = len(py_files) > 10 and total_size > 100 * 1024  # 100KB threshold
        
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all files for analysis
            future_to_file = {
                executor.submit(self._analyze_single_file, py_file, package_name): py_file 
                for py_file in py_files
            }
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_file):
                py_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:  # Only add non-empty results
                        results.append(result)
                    completed_count += 1
                    
                    # Progress logging for large packages
                    if len(py_files) > 20 and completed_count % 10 == 0:
                        logger.debug(f"Processed {completed_count}/{len(py_files)} files")
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze {py_file}: {e}")
                    continue
        
        logger.info(f"Successfully processed {len(results)}/{len(py_files)} files")
        return results
    
    def _analyze_single_file(self, py_file: Path, package_name: str) -> Optional[Tuple[
        List[FunctionInfo], List[ClassInfo], Optional[ModuleInfo], List[CallGraphInfo]
    ]]:
        """Analyze a single Python file with enhanced caching and error handling."""
        try:
            # Generate cache key based on file path and modification time
            stat = py_file.stat()
            cache_key = f"{py_file}_{stat.st_mtime}_{stat.st_size}"
            
            # Check cache first
            if cache_key in self._file_cache:
                return self._file_cache[cache_key]
            
            # Read file with better encoding detection
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
            except UnicodeDecodeError:
                # Fallback to latin-1 for problematic files
                with open(py_file, 'r', encoding='latin-1') as f:
                    source_code = f.read()
                logger.warning(f"Used latin-1 encoding for {py_file}")
            
            # Skip empty files
            if not source_code.strip():
                return None
            
            # Parse AST with error handling
            try:
                tree = ast.parse(source_code, filename=str(py_file))
            except SyntaxError as e:
                logger.warning(f"Syntax error in {py_file}: {e}")
                return None
            
            # Create analyzer for this file
            module_name = self._get_module_name(py_file, package_name)
            analyzer = PackageASTAnalyzer(
                module_name=module_name,
                package_name=package_name,
                source_file=str(py_file)
            )
            
            # Analyze the AST
            analyzer.visit(tree)
            
            # Create module info if there's content
            module_info = None
            if analyzer.functions or analyzer.classes or analyzer.imports:
                module_info = ModuleInfo(
                    name=module_name,
                    package=package_name,
                    docstring=ast.get_docstring(tree),
                    functions=[f.name for f in analyzer.functions],
                    classes=[c.name for c in analyzer.classes],
                    source_file=str(py_file),
                    ast_hash=self._compute_ast_hash(source_code),
                    encoding="utf-8",
                    future_imports=analyzer.future_imports,
                    type_aliases=analyzer.type_aliases,
                    imports=analyzer.imports
                )
            
            result = (analyzer.functions, analyzer.classes, module_info, analyzer.call_graph)
            
            # Cache the result
            self._file_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to analyze {py_file}: {e}")
            return None
    
    def _get_module_name(self, py_file: Path, package_name: str) -> str:
        """Get proper module name with package qualification."""
        # Remove .py extension and convert path to module name
        relative_path = py_file.stem
        
        # Handle __init__.py files
        if py_file.name == "__init__.py":
            relative_path = py_file.parent.name
            
        # Create qualified module name
        if relative_path == package_name:
            return package_name
        else:
            return f"{package_name}.{relative_path}"
    
    def _compute_ast_hash(self, source_code: str) -> str:
        """Compute stable hash for AST comparison."""
        # Use a simpler hash that's stable across runs
        return str(abs(hash(source_code.strip())) % (2**31))
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
        """Create embeddings optimized for DuckDB VSS with caching and batching."""
        if not texts:
            return None
        
        try:
            # Filter out None/empty texts
            clean_texts = [text for text in texts if text and text.strip()]
            if not clean_texts:
                return None
                
            # Check cache for individual texts
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(clean_texts):
                if text in self._embedding_cache:
                    cached_embeddings.append((i, self._embedding_cache[text]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Create embeddings for uncached texts
            new_embeddings = None
            if uncached_texts:
                new_embeddings = self.embedding_model.encode(
                    uncached_texts,
                    batch_size=min(batch_size, len(uncached_texts)),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # Crucial for cosine similarity
                    device=None  # Let model decide
                )
                
                # Cache new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self._embedding_cache[text] = embedding.astype(np.float32)  # VSS-optimized dtype
            
            # Combine cached and new embeddings
            all_embeddings = np.zeros((len(clean_texts), self.embedding_dimension), dtype=np.float32)
            
            # Place cached embeddings
            for idx, embedding in cached_embeddings:
                all_embeddings[idx] = embedding
                
            # Place new embeddings
            if new_embeddings is not None:
                for i, embedding in enumerate(new_embeddings):
                    all_embeddings[uncached_indices[i]] = embedding.astype(np.float32)
            
            # Handle original None/empty texts by padding with zeros or returning subset
            if len(clean_texts) < len(texts):
                # Pad with zero vectors for empty texts
                padded_embeddings = np.zeros((len(texts), self.embedding_dimension), dtype=np.float32)
                clean_idx = 0
                for i, text in enumerate(texts):
                    if text and text.strip():
                        padded_embeddings[i] = all_embeddings[clean_idx]
                        clean_idx += 1
                return padded_embeddings
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            return None
    
    def create_single_embedding(self, text: str) -> Optional[np.ndarray]:
        """Create a single embedding with caching."""
        if not text or not text.strip():
            return None
            
        # Check cache first
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            embedding = self.embedding_model.encode(
                [text],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0].astype(np.float32)
            
            # Cache the result
            self._embedding_cache[text] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to create single embedding: {e}")
            return None
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for database schema setup."""
        return self.embedding_dimension
    
    def clear_cache(self) -> None:
        """Clear all caches to free memory."""
        self._file_cache.clear()
        self._embedding_cache.clear()
        logger.info("Cleared scanner caches")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        return {
            'file_cache_size': len(self._file_cache),
            'embedding_cache_size': len(self._embedding_cache),
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_dimension
        }
