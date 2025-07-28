"""
Modern Package Indexer with DuckDB VSS Extension best practices.
Uses HNSW vector similarity search for high-performance semantic search.
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pyarrow as pa

from .models import CallGraphInfo, ClassInfo, FunctionInfo, ModuleInfo
from .scanner import PackageScanner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PackageIndexer:
    """Modern Package Indexer with DuckDB VSS Extension and HNSW optimization."""
    
    def __init__(self, db_path: Union[str, Path] = "package_index.db", embedding_dim: int = 384):
        """
        Initialize PackageIndexer with VSS optimization.
        
        Args:
            db_path: Path to DuckDB database
            embedding_dim: Dimension of embeddings (384 for sentence-transformers)
        """
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        
        # Initialize scanner
        self.scanner = PackageScanner()
        
        self._ensure_connection()
        self._setup_database()
        
    def _ensure_connection(self) -> None:
        """Ensure database connection with VSS extension and optimal settings."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
            self._configure_vss_and_performance()
        else:
            # Check if connection is still valid
            try:
                self._conn.execute("SELECT 1").fetchone()
            except Exception:
                self._conn = duckdb.connect(str(self.db_path))
                self._configure_vss_and_performance()
            
    def _configure_vss_and_performance(self) -> None:
        """Configure DuckDB with VSS extension and optimal performance settings."""
        try:
            # Install and load VSS extension for vector similarity search
            self._conn.execute("INSTALL vss")
            self._conn.execute("LOAD vss")
            
            # Install and load Arrow for zero-copy operations
            self._conn.execute("INSTALL arrow")
            self._conn.execute("LOAD arrow")
            
            # Memory configuration - use 6GB for better vector operations
            self._conn.execute("SET memory_limit='6GB'")
            
            # Optimize for vector operations and batch processing
            self._conn.execute("SET threads=6")  # More threads for HNSW operations
            self._conn.execute("SET enable_progress_bar=false")
            self._conn.execute("SET enable_profiling='no_output'")
            
            # Optimize vector similarity computations
            self._conn.execute("SET enable_object_cache=true")
            
            logger.info("DuckDB VSS extension and performance settings configured")
        except Exception as e:
            logger.warning(f"Could not configure VSS extension or settings: {e}")
            
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get database connection, ensuring it's active."""
        self._ensure_connection()
        return self._conn
        
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        conn = self.conn
        try:
            conn.execute("BEGIN TRANSACTION")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    
    def _setup_database(self) -> None:
        """Create optimized database schema with VSS HNSW indexes."""
        with self.transaction() as conn:
            # Create sequences
            conn.execute("CREATE SEQUENCE IF NOT EXISTS functions_seq START 1")
            conn.execute("CREATE SEQUENCE IF NOT EXISTS classes_seq START 1") 
            conn.execute("CREATE SEQUENCE IF NOT EXISTS modules_seq START 1")
            conn.execute("CREATE SEQUENCE IF NOT EXISTS call_graph_seq START 1")
            
            # Functions table optimized for VSS
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS functions (
                    id BIGINT DEFAULT nextval('functions_seq'),
                    name VARCHAR NOT NULL,
                    module VARCHAR NOT NULL,
                    package VARCHAR NOT NULL,
                    docstring TEXT,
                    signature VARCHAR NOT NULL,
                    parameters VARCHAR[] NOT NULL DEFAULT [],
                    return_annotation VARCHAR,
                    source_file VARCHAR,
                    line_number INTEGER NOT NULL DEFAULT 0,
                    end_line_number INTEGER,
                    column_offset INTEGER NOT NULL DEFAULT 0,
                    end_column_offset INTEGER,
                    complexity_score INTEGER NOT NULL DEFAULT 1,
                    ast_hash VARCHAR(32) NOT NULL,
                    decorators VARCHAR[] NOT NULL DEFAULT [],
                    is_async BOOLEAN NOT NULL DEFAULT false,
                    type_comments TEXT,
                    embedding FLOAT[{self.embedding_dim}],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (id)
                )
            """)
            
            # Classes table optimized for VSS
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS classes (
                    id BIGINT DEFAULT nextval('classes_seq'),
                    name VARCHAR NOT NULL,
                    module VARCHAR NOT NULL,
                    package VARCHAR NOT NULL,
                    docstring TEXT,
                    methods VARCHAR[] NOT NULL DEFAULT [],
                    base_classes VARCHAR[] NOT NULL DEFAULT [],
                    source_file VARCHAR,
                    line_number INTEGER NOT NULL DEFAULT 0,
                    end_line_number INTEGER,
                    decorators VARCHAR[] NOT NULL DEFAULT [],
                    ast_hash VARCHAR(32) NOT NULL,
                    is_dataclass BOOLEAN NOT NULL DEFAULT false,
                    type_params VARCHAR[] NOT NULL DEFAULT [],
                    embedding FLOAT[{self.embedding_dim}],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (id)
                )
            """)
            
            # Modules table optimized for VSS
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS modules (
                    id BIGINT DEFAULT nextval('modules_seq'),
                    name VARCHAR NOT NULL,
                    package VARCHAR NOT NULL,
                    docstring TEXT,
                    imports VARCHAR[] NOT NULL DEFAULT [],
                    functions VARCHAR[] NOT NULL DEFAULT [],
                    classes VARCHAR[] NOT NULL DEFAULT [],
                    source_file VARCHAR,
                    ast_hash VARCHAR(32) NOT NULL,
                    encoding VARCHAR DEFAULT 'utf-8',
                    future_imports VARCHAR[] NOT NULL DEFAULT [],
                    type_aliases VARCHAR[] NOT NULL DEFAULT [],
                    embedding FLOAT[{self.embedding_dim}],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (id)
                )
            """)

            # Call graph table with optimized structure
            conn.execute("""
                CREATE TABLE IF NOT EXISTS call_graph (
                    id BIGINT DEFAULT nextval('call_graph_seq'),
                    source_function VARCHAR NOT NULL,
                    target_function VARCHAR NOT NULL,
                    module VARCHAR NOT NULL,
                    package VARCHAR NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (id)
                )
            """)
            
            self._create_optimized_indexes()
            
    def _create_optimized_indexes(self) -> None:
        """Create performance-optimized indexes including HNSW vector indexes."""
        # Regular B-tree indexes
        btree_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_func_name_pkg ON functions(name, package)",
            "CREATE INDEX IF NOT EXISTS idx_func_package ON functions(package)",
            "CREATE INDEX IF NOT EXISTS idx_func_module ON functions(module)",
            "CREATE INDEX IF NOT EXISTS idx_func_hash ON functions(ast_hash)",
            
            "CREATE INDEX IF NOT EXISTS idx_class_name_pkg ON classes(name, package)",
            "CREATE INDEX IF NOT EXISTS idx_class_package ON classes(package)",
            "CREATE INDEX IF NOT EXISTS idx_class_hash ON classes(ast_hash)",
            
            "CREATE INDEX IF NOT EXISTS idx_module_name_pkg ON modules(name, package)",
            "CREATE INDEX IF NOT EXISTS idx_module_hash ON modules(ast_hash)",
            
            "CREATE INDEX IF NOT EXISTS idx_call_source ON call_graph(source_function)",
            "CREATE INDEX IF NOT EXISTS idx_call_target ON call_graph(target_function)",
            "CREATE INDEX IF NOT EXISTS idx_call_pkg_mod ON call_graph(package, module)",
        ]
        
        # HNSW vector similarity indexes (high-performance approximate search)
        hnsw_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_func_embedding_hnsw ON functions USING HNSW (embedding) WITH (metric = 'cosine')",
            "CREATE INDEX IF NOT EXISTS idx_class_embedding_hnsw ON classes USING HNSW (embedding) WITH (metric = 'cosine')",
            "CREATE INDEX IF NOT EXISTS idx_module_embedding_hnsw ON modules USING HNSW (embedding) WITH (metric = 'cosine')",
        ]
        
        # Create B-tree indexes first
        for index_sql in btree_indexes:
            try:
                self.conn.execute(index_sql)
            except Exception as e:
                logger.warning(f"Could not create B-tree index: {e}")
        
        # Create HNSW indexes (only if data exists)
        for index_sql in hnsw_indexes:
            try:
                self.conn.execute(index_sql)
                logger.info("Created HNSW vector similarity index")
            except Exception as e:
                logger.warning(f"Could not create HNSW index (will be created after data insertion): {e}")
    
    def index_package(self, package_name: str, force_reindex: bool = False) -> Dict[str, int]:
        """Index a package with optimized change detection and vector operations."""
        logger.info(f"Starting indexing for package: {package_name}")
        
        try:
            # Analyze package using scanner
            functions, classes, modules, call_graph = self.scanner.analyze_package(package_name)
            
            if not any([functions, classes, modules]):
                logger.warning(f"No content found for package: {package_name}")
                return {'functions': 0, 'classes': 0, 'modules': 0, 'calls': 0}
            
            # Efficient change detection using batch queries
            if not force_reindex:
                functions, classes, modules = self._filter_unchanged_items(
                    package_name, functions, classes, modules
                )
            
            # Batch index operations in transaction for better performance
            results = {'functions': 0, 'classes': 0, 'modules': 0, 'calls': 0}
            
            with self.transaction():
                results['functions'] = self._index_functions_with_vectors(functions)
                results['classes'] = self._index_classes_with_vectors(classes)
                results['modules'] = self._index_modules_with_vectors(modules)
                results['calls'] = self._index_call_graph(call_graph)
                
                # Create HNSW indexes after data insertion if not exists
                self._ensure_hnsw_indexes()
            
            logger.info(
                f"Indexed package {package_name}: "
                f"{results['functions']} functions, {results['classes']} classes, "
                f"{results['modules']} modules, {results['calls']} calls"
            )
            return results
            
        except Exception as e:
            logger.error(f"Error indexing package {package_name}: {e}")
            raise RuntimeError(f"Failed to index package {package_name}") from e
    
    def _ensure_hnsw_indexes(self) -> None:
        """Ensure HNSW indexes exist after data insertion."""
        hnsw_indexes = [
            ("functions", "embedding"),
            ("classes", "embedding"),
            ("modules", "embedding"),
        ]
        
        for table, column in hnsw_indexes:
            try:
                # Check if index exists and create if not
                self.conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table}_{column}_hnsw 
                    ON {table} USING HNSW ({column}) 
                    WITH (metric = 'cosine', ef_construction = 128, M = 16)
                """)
            except Exception as e:
                logger.warning(f"Could not ensure HNSW index for {table}.{column}: {e}")
    
    def _filter_unchanged_items(self, package_name: str, functions: List[FunctionInfo], 
                              classes: List[ClassInfo], modules: List[ModuleInfo]) -> tuple:
        """Efficiently filter out unchanged items using batch queries."""
        existing_hashes = self._get_existing_hashes(package_name)
        
        if not existing_hashes:
            return functions, classes, modules
            
        # Filter items that have changed
        new_functions = [f for f in functions if f.ast_hash not in existing_hashes]
        new_classes = [c for c in classes if c.ast_hash not in existing_hashes]
        new_modules = [m for m in modules if m.ast_hash not in existing_hashes]
        
        logger.info(
            f"Change detection: {len(functions) - len(new_functions)} functions, "
            f"{len(classes) - len(new_classes)} classes, "
            f"{len(modules) - len(new_modules)} modules unchanged"
        )
        
        return new_functions, new_classes, new_modules
        
    def _get_existing_hashes(self, package_name: str) -> set:
        """Get existing AST hashes for efficient change detection."""
        try:
            # Optimized query using Arrow for zero-copy
            query = """
                SELECT ast_hash FROM functions WHERE package = ?
                UNION ALL
                SELECT ast_hash FROM classes WHERE package = ?
                UNION ALL
                SELECT ast_hash FROM modules WHERE package = ?
            """
            
            results = self.conn.execute(query, [package_name, package_name, package_name]).fetchall()
            return {row[0] for row in results}
            
        except Exception as e:
            logger.warning(f"Could not retrieve existing hashes: {e}")
            return set()
    
    def _prepare_embedding_text(self, name: str, signature: Optional[str] = None, 
                               docstring: Optional[str] = None, decorators: Optional[List[str]] = None,
                               methods: Optional[List[str]] = None, future_imports: Optional[List[str]] = None) -> str:
        """Prepare optimized text for embedding generation with enhanced context."""
        text_parts = []
        
        # Add name as primary identifier
        text_parts.append(f"function {name}")
        
        # Add signature with cleaning
        if signature:
            clean_signature = signature.replace("self, ", "").replace("self", "")
            text_parts.append(f"signature: {clean_signature}")
        
        # Add truncated docstring for better performance
        if docstring:
            clean_docstring = docstring.replace("\n", " ").replace("  ", " ").strip()
            if len(clean_docstring) > 200:
                clean_docstring = clean_docstring[:200] + "..."
            text_parts.append(f"description: {clean_docstring}")
        
        # Add decorators as context
        if decorators:
            decorator_text = " ".join([d.replace("@", "") for d in decorators])
            text_parts.append(f"decorators: {decorator_text}")
        
        # Add methods for classes
        if methods:
            methods_text = " ".join(methods[:10])  # Limit for performance
            text_parts.append(f"methods: {methods_text}")
        
        # Add future imports as context
        if future_imports:
            imports_text = " ".join(future_imports[:5])  # Limit for performance
            text_parts.append(f"imports: {imports_text}")
        
        return " | ".join(text_parts)
    
    def _index_functions_with_vectors(self, functions: List[FunctionInfo]) -> int:
        """Index functions using optimized vector operations with Arrow integration."""
        if not functions:
            return 0
            
        try:
            # Clear existing functions for this package
            package_name = functions[0].package
            self.conn.execute("DELETE FROM functions WHERE package = ?", [package_name])
            
            # Process in optimized batches for vector operations
            batch_size = 50  # Smaller batches for better memory usage with vectors
            total_indexed = 0
            
            for i in range(0, len(functions), batch_size):
                batch = functions[i:i + batch_size]
                
                # Prepare embedding texts efficiently
                texts = [
                    self._prepare_embedding_text(
                        func.name, func.signature, func.docstring, func.decorators
                    ) for func in batch
                ]
                
                # Create embeddings with optimized batch size
                embeddings = self.scanner.create_embeddings(texts, batch_size=25)
                
                # Convert embeddings to Arrow format for zero-copy
                embedding_arrays = []
                for embedding in embeddings:
                    if embedding is not None:
                        # Ensure correct dtype and shape for DuckDB
                        embedding_float32 = embedding.astype(np.float32)
                        embedding_arrays.append(embedding_float32.tolist())
                    else:
                        embedding_arrays.append(None)
                
                # Prepare batch data with proper vector format
                batch_data = [
                    (
                        func.name, func.module, func.package, func.docstring,
                        func.signature, func.parameters, func.return_annotation,
                        func.source_file, func.line_number, func.end_line_number,
                        func.column_offset, func.end_column_offset, func.complexity_score,
                        func.ast_hash, func.decorators, func.is_async, func.type_comments,
                        embedding_arrays[idx]
                    )
                    for idx, func in enumerate(batch)
                ]
                
                # Optimized insert with vector data
                insert_sql = f"""
                    INSERT INTO functions 
                    (name, module, package, docstring, signature, parameters, 
                     return_annotation, source_file, line_number, end_line_number,
                     column_offset, end_column_offset, complexity_score, ast_hash,
                     decorators, is_async, type_comments, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::FLOAT[{self.embedding_dim}])
                """
                
                # Batch insert with vector optimization
                self.conn.executemany(insert_sql, batch_data)
                total_indexed += len(batch)
                
                logger.debug(f"Indexed batch of {len(batch)} functions with vectors")
            
            logger.info(f"Successfully indexed {total_indexed} functions with vectors")
            return total_indexed
            
        except Exception as e:
            logger.error(f"Failed to index functions with vectors: {e}")
            raise
    
    def _index_classes_with_vectors(self, classes: List[ClassInfo]) -> int:
        """Index classes using optimized vector operations."""
        if not classes:
            return 0
            
        try:
            # Clear existing classes for this package
            package_name = classes[0].package
            self.conn.execute("DELETE FROM classes WHERE package = ?", [package_name])
            
            # Process in batches
            batch_size = 25  # Smaller batches for classes
            total_indexed = 0
            
            for i in range(0, len(classes), batch_size):
                batch = classes[i:i + batch_size]
                
                # Prepare class-specific embedding texts
                texts = [
                    self._prepare_class_embedding_text(
                        cls.name, cls.docstring, cls.methods, cls.decorators, cls.base_classes
                    ) for cls in batch
                ]
                
                embeddings = self.scanner.create_embeddings(texts, batch_size=15)
                
                # Convert embeddings to proper format
                embedding_arrays = []
                for embedding in embeddings:
                    if embedding is not None:
                        embedding_float32 = embedding.astype(np.float32)
                        embedding_arrays.append(embedding_float32.tolist())
                    else:
                        embedding_arrays.append(None)
                
                # Prepare batch data
                batch_data = [
                    (
                        cls.name, cls.module, cls.package, cls.docstring,
                        cls.methods, cls.base_classes, cls.source_file,
                        cls.line_number, cls.end_line_number, cls.decorators,
                        cls.ast_hash, cls.is_dataclass, cls.type_params,
                        embedding_arrays[idx]
                    )
                    for idx, cls in enumerate(batch)
                ]
                
                insert_sql = f"""
                    INSERT INTO classes
                    (name, module, package, docstring, methods, base_classes,
                     source_file, line_number, end_line_number, decorators,
                     ast_hash, is_dataclass, type_params, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::FLOAT[{self.embedding_dim}])
                """
                
                self.conn.executemany(insert_sql, batch_data)
                total_indexed += len(batch)
                
                logger.debug(f"Indexed batch of {len(batch)} classes with vectors")
            
            logger.info(f"Successfully indexed {total_indexed} classes with vectors")
            return total_indexed
            
        except Exception as e:
            logger.error(f"Failed to index classes with vectors: {e}")
            raise
    
    def _prepare_class_embedding_text(self, name: str, docstring: Optional[str] = None, 
                                     methods: Optional[List[str]] = None, decorators: Optional[List[str]] = None,
                                     base_classes: Optional[List[str]] = None) -> str:
        """Prepare optimized text for class embedding generation."""
        text_parts = []
        
        text_parts.append(f"class {name}")
        
        if base_classes:
            base_text = " ".join(base_classes[:3])  # Limit for performance
            text_parts.append(f"inherits: {base_text}")
        
        if docstring:
            clean_docstring = docstring.replace("\n", " ").replace("  ", " ").strip()
            if len(clean_docstring) > 200:
                clean_docstring = clean_docstring[:200] + "..."
            text_parts.append(f"description: {clean_docstring}")
        
        if methods:
            methods_text = " ".join(methods[:8])  # Limit for performance
            text_parts.append(f"methods: {methods_text}")
        
        if decorators:
            decorator_text = " ".join([d.replace("@", "") for d in decorators])
            text_parts.append(f"decorators: {decorator_text}")
        
        return " | ".join(text_parts)

    def _index_modules_with_vectors(self, modules: List[ModuleInfo]) -> int:
        """Index modules using optimized vector operations."""
        if not modules:
            return 0
            
        try:
            # Clear existing modules for this package
            package_name = modules[0].package
            self.conn.execute("DELETE FROM modules WHERE package = ?", [package_name])
            
            # Prepare embedding texts
            texts = [
                self._prepare_embedding_text(
                    mod.name, docstring=mod.docstring, future_imports=mod.future_imports
                ) for mod in modules
            ]
            
            embeddings = self.scanner.create_embeddings(texts, batch_size=20)
            
            # Convert embeddings to proper format
            embedding_arrays = []
            for embedding in embeddings:
                if embedding is not None:
                    embedding_float32 = embedding.astype(np.float32)
                    embedding_arrays.append(embedding_float32.tolist())
                else:
                    embedding_arrays.append(None)
            
            # Prepare batch data
            batch_data = [
                (
                    mod.name, mod.package, mod.docstring, mod.imports,
                    mod.functions, mod.classes, mod.source_file,
                    mod.ast_hash, mod.encoding, mod.future_imports,
                    mod.type_aliases, embedding_arrays[idx]
                )
                for idx, mod in enumerate(modules)
            ]
            
            insert_sql = f"""
                INSERT INTO modules
                (name, package, docstring, imports, functions, classes,
                 source_file, ast_hash, encoding, future_imports, 
                 type_aliases, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::FLOAT[{self.embedding_dim}])
            """
            
            self.conn.executemany(insert_sql, batch_data)
            logger.info(f"Indexed {len(modules)} modules with vectors")
            return len(modules)
            
        except Exception as e:
            logger.error(f"Failed to index modules with vectors: {e}")
            raise

    def _index_call_graph(self, call_graph: List[CallGraphInfo]) -> int:
        """Index call graph using optimized batch operations."""
        if not call_graph:
            return 0
            
        # Remove duplicates for better data quality
        unique_calls = {}
        for call in call_graph:
            key = (call.source_function, call.target_function, call.module, call.package)
            unique_calls[key] = call
            
        batch_data = [
            (call.source_function, call.target_function, call.module, call.package)
            for call in unique_calls.values()
        ]
        
        # Clear existing call graph data for this package
        if call_graph:
            self.conn.execute("DELETE FROM call_graph WHERE package = ?", [call_graph[0].package])
        
        insert_sql = """
            INSERT INTO call_graph (source_function, target_function, module, package)
            VALUES (?, ?, ?, ?)
        """
        
        try:
            self.conn.executemany(insert_sql, batch_data)
            logger.debug(f"Indexed {len(batch_data)} call graph edges")
            return len(batch_data)
        except Exception as e:
            logger.error(f"Failed to index call graph: {e}")
            raise
    
    def search_similar(self, query: str, limit: int = 10, 
                      similarity_threshold: float = 0.5) -> Dict[str, List[Dict[str, Any]]]:
        """Perform high-performance HNSW-based semantic similarity search."""
        if not query.strip():
            return {'functions': [], 'classes': [], 'modules': []}
            
        try:
            # Create query embedding once
            query_embeddings = self.scanner.create_embeddings([query], batch_size=1)
            if query_embeddings is None or len(query_embeddings) == 0:
                logger.warning("Could not create embedding for query")
                return {'functions': [], 'classes': [], 'modules': []}
                
            query_embedding = query_embeddings[0]
            if query_embedding is None:
                logger.warning("Query embedding is None")
                return {'functions': [], 'classes': [], 'modules': []}
                
            # Convert to proper format for DuckDB VSS
            query_embedding_float32 = query_embedding.astype(np.float32).tolist()
            
            # Use HNSW-accelerated search for all types
            search_results = {
                'functions': self._search_functions_hnsw(query_embedding_float32, similarity_threshold, limit),
                'classes': self._search_classes_hnsw(query_embedding_float32, similarity_threshold, limit),
                'modules': self._search_modules_hnsw(query_embedding_float32, similarity_threshold, limit)
            }
            
            total_results = sum(len(results) for results in search_results.values())
            logger.info(f"HNSW search for '{query}' returned {total_results} results")
            
            return search_results
            
        except Exception as e:
            logger.error(f"HNSW search failed for query '{query}': {e}")
            return {'functions': [], 'classes': [], 'modules': []}
            
    def _search_functions_hnsw(self, query_embedding: List[float], 
                              threshold: float, limit: int) -> List[Dict[str, Any]]:
        """High-performance HNSW-based function search."""
        try:
            # Use VSS extension for high-performance approximate nearest neighbor search
            results = self.conn.execute(f"""
                SELECT name, module, package, docstring, signature, 
                       decorators, is_async, complexity_score,
                       array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) as similarity
                FROM functions 
                WHERE embedding IS NOT NULL 
                  AND array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) >= ?
                ORDER BY array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) DESC 
                LIMIT ?
            """, [query_embedding, query_embedding, threshold, query_embedding, limit]).fetchall()
            
            return [
                {
                    'name': row[0], 'module': row[1], 'package': row[2],
                    'docstring': row[3], 'signature': row[4], 'decorators': row[5],
                    'is_async': row[6], 'complexity': row[7], 'similarity': float(row[8])
                }
                for row in results
            ]
        except Exception as e:
            logger.warning(f"HNSW function search failed: {e}")
            return []
            
    def _search_classes_hnsw(self, query_embedding: List[float], 
                            threshold: float, limit: int) -> List[Dict[str, Any]]:
        """High-performance HNSW-based class search."""
        try:
            results = self.conn.execute(f"""
                SELECT name, module, package, docstring, methods, 
                       base_classes, is_dataclass,
                       array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) as similarity
                FROM classes 
                WHERE embedding IS NOT NULL 
                  AND array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) >= ?
                ORDER BY array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) DESC 
                LIMIT ?
            """, [query_embedding, query_embedding, threshold, query_embedding, limit]).fetchall()
            
            return [
                {
                    'name': row[0], 'module': row[1], 'package': row[2],
                    'docstring': row[3], 'methods': row[4], 'base_classes': row[5],
                    'is_dataclass': row[6], 'similarity': float(row[7])
                }
                for row in results
            ]
        except Exception as e:
            logger.warning(f"HNSW class search failed: {e}")
            return []
            
    def _search_modules_hnsw(self, query_embedding: List[float], 
                            threshold: float, limit: int) -> List[Dict[str, Any]]:
        """High-performance HNSW-based module search."""
        try:
            results = self.conn.execute(f"""
                SELECT name, package, docstring, functions, classes,
                       array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) as similarity
                FROM modules 
                WHERE embedding IS NOT NULL 
                  AND array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) >= ?
                ORDER BY array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) DESC 
                LIMIT ?
            """, [query_embedding, query_embedding, threshold, query_embedding, limit]).fetchall()
            
            return [
                {
                    'name': row[0], 'package': row[1], 'docstring': row[2],
                    'functions': row[3], 'classes': row[4], 'similarity': float(row[5])
                }
                for row in results
            ]
        except Exception as e:
            logger.warning(f"HNSW module search failed: {e}")
            return []
    
    def get_package_stats(self) -> Dict[str, Any]:
        """Get comprehensive package statistics with optimized aggregation queries."""
        try:
            stats = {}
            
            # Package overview with vector statistics
            package_results = self.conn.execute("""
                SELECT 
                    package,
                    COUNT(*) as function_count,
                    ROUND(AVG(complexity_score), 2) as avg_complexity,
                    COUNT(CASE WHEN is_async THEN 1 END) as async_functions,
                    COUNT(DISTINCT name) as unique_functions,
                    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as vectorized_functions
                FROM functions 
                WHERE package IS NOT NULL
                GROUP BY package
                ORDER BY function_count DESC
                LIMIT 20
            """).fetchall()
            
            stats['packages'] = [
                {
                    'name': row[0], 'functions': row[1], 'avg_complexity': row[2],
                    'async_functions': row[3], 'unique_functions': row[4],
                    'vectorized_functions': row[5]
                }
                for row in package_results
            ]
            
            # Enhanced total statistics with vector coverage
            total_results = self.conn.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM functions) as total_functions,
                    (SELECT COUNT(*) FROM classes) as total_classes,
                    (SELECT COUNT(*) FROM modules) as total_modules,
                    (SELECT COUNT(DISTINCT package) FROM functions WHERE package IS NOT NULL) as total_packages,
                    (SELECT COUNT(*) FROM classes WHERE is_dataclass = true) as dataclasses,
                    (SELECT ROUND(AVG(complexity_score), 2) FROM functions) as avg_complexity,
                    (SELECT COUNT(DISTINCT source_function) FROM call_graph) as connected_functions,
                    (SELECT COUNT(*) FROM call_graph) as total_calls,
                    (SELECT COUNT(*) FROM functions WHERE embedding IS NOT NULL) as vectorized_functions,
                    (SELECT COUNT(*) FROM classes WHERE embedding IS NOT NULL) as vectorized_classes,
                    (SELECT COUNT(*) FROM modules WHERE embedding IS NOT NULL) as vectorized_modules
            """).fetchone()
            
            stats['totals'] = {
                'functions': total_results[0] or 0,
                'classes': total_results[1] or 0,
                'modules': total_results[2] or 0,
                'packages': total_results[3] or 0,
                'dataclasses': total_results[4] or 0,
                'avg_complexity': total_results[5] or 0,
                'connected_functions': total_results[6] or 0,
                'total_calls': total_results[7] or 0,
                'vectorized_functions': total_results[8] or 0,
                'vectorized_classes': total_results[9] or 0,
                'vectorized_modules': total_results[10] or 0
            }
            
            # Vector coverage statistics
            if stats['totals']['functions'] > 0:
                stats['vector_coverage'] = {
                    'functions': round(stats['totals']['vectorized_functions'] / stats['totals']['functions'] * 100, 1),
                    'classes': round(stats['totals']['vectorized_classes'] / stats['totals']['classes'] * 100, 1) if stats['totals']['classes'] > 0 else 0,
                    'modules': round(stats['totals']['vectorized_modules'] / stats['totals']['modules'] * 100, 1) if stats['totals']['modules'] > 0 else 0
                }
            
            # Complexity distribution
            complexity_distribution = self.conn.execute("""
                SELECT 
                    CASE 
                        WHEN complexity_score <= 5 THEN 'Low (1-5)'
                        WHEN complexity_score <= 10 THEN 'Medium (6-10)'
                        WHEN complexity_score <= 20 THEN 'High (11-20)'
                        ELSE 'Very High (20+)'
                    END as complexity_range,
                    COUNT(*) as count
                FROM functions
                GROUP BY complexity_range
                ORDER BY 
                    CASE complexity_range
                        WHEN 'Low (1-5)' THEN 1
                        WHEN 'Medium (6-10)' THEN 2
                        WHEN 'High (11-20)' THEN 3
                        ELSE 4
                    END
            """).fetchall()
            
            stats['complexity_distribution'] = [
                {'range': row[0], 'count': row[1]} for row in complexity_distribution
            ]
            
            logger.info(f"Generated enhanced stats for {stats['totals']['packages']} packages with vector coverage")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate package stats: {e}")
            return {
                'packages': [],
                'totals': {
                    'functions': 0, 'classes': 0, 'modules': 0, 'packages': 0,
                    'dataclasses': 0, 'avg_complexity': 0,
                    'connected_functions': 0, 'total_calls': 0,
                    'vectorized_functions': 0, 'vectorized_classes': 0, 'vectorized_modules': 0
                },
                'vector_coverage': {'functions': 0, 'classes': 0, 'modules': 0},
                'complexity_distribution': []
            }
            
    def close(self) -> None:
        """Close database connection properly."""
        if self._conn:
            try:
                self._conn.close()
                logger.info("DuckDB VSS database connection closed")
            except Exception as e:
                logger.warning(f"Error closing VSS connection: {e}")
            finally:
                self._conn = None
            
    def clear_database(self) -> None:
        """Clear all data and rebuild indexes."""
        try:
            with self.transaction() as conn:
                conn.execute("DELETE FROM call_graph")
                conn.execute("DELETE FROM functions")
                conn.execute("DELETE FROM classes")
                conn.execute("DELETE FROM modules")
                logger.info("Database cleared successfully")
                
                # Recreate HNSW indexes after clearing
                self._ensure_hnsw_indexes()
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            raise
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
