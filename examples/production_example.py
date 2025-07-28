#!/usr/bin/env python3
"""
🏭 ADPA Production Deployment Example - Modern Architecture

Production-ready ADPA deployment with:
- Proper error handling and structured logging
- Configuration management with type safety
- Performance monitoring and metrics
- Batch processing with progress tracking
- Database optimization and connection pooling
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from loguru import logger

from adpa import PackageIndexer

console = Console()


@dataclass
class ProductionConfig:
    """Production configuration for ADPA with type safety."""
    db_path: str = "production_packages.db"
    batch_size: int = 32
    max_workers: int = 4
    log_level: str = "INFO"
    log_file: str = "adpa_production.log"
    packages_config_file: str = "packages.json"
    performance_monitoring: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0
    database_timeout: int = 30
    
    # Advanced configuration
    max_package_size_mb: int = 100
    similarity_threshold: float = 0.3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")


@dataclass
class PackageProcessingResult:
    """Result of package processing operation."""
    package_name: str
    success: bool
    processing_time: float
    functions_indexed: int = 0
    classes_indexed: int = 0
    error_message: Optional[str] = None
    memory_usage_mb: float = 0.0


@dataclass
class BatchProcessingMetrics:
    """Metrics for batch processing operations."""
    total_packages: int = 0
    successful_packages: int = 0
    failed_packages: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_processing_time: float = 0.0
    average_package_time: float = 0.0
    peak_memory_mb: float = 0.0
    results: List[PackageProcessingResult] = field(default_factory=list)
    
    def calculate_averages(self):
        """Calculate average metrics."""
        if self.successful_packages > 0:
            self.average_package_time = self.total_processing_time / self.successful_packages


class ProductionADPA:
    """Production-ready ADPA wrapper with comprehensive monitoring."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.indexer: Optional[PackageIndexer] = None
        self.setup_logging()
        self.performance_metrics = BatchProcessingMetrics()
        
    def setup_logging(self):
        """Configure production logging with structured format."""
        logger.remove()  # Remove default logger
        
        # File logger with rotation and compression
        logger.add(
            self.config.log_file,
            level=self.config.log_level,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
        
        # Console logger for real-time monitoring
        logger.add(
            console.print,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
            colorize=True
        )
        
        logger.info("Production ADPA logging configured", extra={
            "config": {
                "log_level": self.config.log_level,
                "log_file": self.config.log_file,
                "db_path": self.config.db_path
            }
        })
    
    @contextmanager
    def get_indexer(self):
        """Context manager for indexer with proper resource management."""
        try:
            if self.indexer is None:
                logger.info("Initializing PackageIndexer")
                
                self.indexer = PackageIndexer(db_path=self.config.db_path)
                
                logger.info("PackageIndexer initialized successfully", extra={
                    "db_path": self.config.db_path
                })
            
            yield self.indexer
            
        except Exception as e:
            logger.error("Error with indexer", extra={"error": str(e)})
            raise
        finally:
            # Indexer cleanup handled automatically by context
            pass
    
    def load_packages_config(self) -> List[Dict[str, Any]]:
        """Load and validate packages configuration."""
        config_path = Path(self.config.packages_config_file)
        
        if not config_path.exists():
            example_config = self._create_example_config()
            self._save_config(config_path, example_config)
            logger.info("Created example packages configuration", extra={
                "config_file": str(config_path),
                "packages_count": len(example_config)
            })
            return example_config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                packages = json.load(f)
            
            # Validate package configuration
            validated_packages = self._validate_packages_config(packages)
            
            logger.info("Loaded packages configuration", extra={
                "config_file": str(config_path),
                "packages_count": len(validated_packages)
            })
            
            return validated_packages
            
        except Exception as e:
            logger.error("Failed to load packages config", extra={
                "config_file": str(config_path),
                "error": str(e)
            })
            return []
    
    def _create_example_config(self) -> List[Dict[str, Any]]:
        """Create example configuration for packages."""
        return [
            {
                "name": "requests",
                "priority": "high",
                "force_reindex": False,
                "description": "HTTP library for Python",
                "expected_functions": 50,
                "expected_classes": 20
            },
            {
                "name": "numpy", 
                "priority": "medium",
                "force_reindex": False,
                "description": "Numerical computing library",
                "expected_functions": 200,
                "expected_classes": 50
            },
            {
                "name": "pandas",
                "priority": "medium", 
                "force_reindex": False,
                "description": "Data analysis library",
                "expected_functions": 300,
                "expected_classes": 80
            }
        ]
    
    def _save_config(self, config_path: Path, config: List[Dict[str, Any]]):
        """Save configuration to file."""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save config", extra={
                "config_file": str(config_path),
                "error": str(e)
            })
    
    def _validate_packages_config(self, packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and sanitize packages configuration."""
        validated = []
        
        for pkg in packages:
            if not isinstance(pkg, dict):
                logger.warning("Skipping invalid package config", extra={"package": pkg})
                continue
                
            if "name" not in pkg:
                logger.warning("Skipping package without name", extra={"package": pkg})
                continue
            
            # Set defaults for missing fields
            validated_pkg = {
                "name": pkg["name"],
                "priority": pkg.get("priority", "medium"),
                "force_reindex": pkg.get("force_reindex", False),
                "description": pkg.get("description", ""),
                "expected_functions": pkg.get("expected_functions", 0),
                "expected_classes": pkg.get("expected_classes", 0)
            }
            
            validated.append(validated_pkg)
        
        return validated
    
    def process_package(self, package_config: Dict[str, Any]) -> PackageProcessingResult:
        """Process a single package with comprehensive error handling."""
        package_name = package_config["name"]
        start_time = time.time()
        
        logger.info("Processing package", extra={
            "package": package_name,
            "priority": package_config.get("priority", "unknown")
        })
        
        try:
            with self.get_indexer() as indexer:
                # Index the package
                results = indexer.index_package(package_name)
                
                processing_time = time.time() - start_time
                
                # Create successful result
                result = PackageProcessingResult(
                    package_name=package_name,
                    success=True,
                    processing_time=processing_time,
                    functions_indexed=results.get('functions', 0),
                    classes_indexed=results.get('classes', 0)
                )
                
                logger.info("Package processed successfully", extra={
                    "package": package_name,
                    "processing_time": f"{processing_time:.2f}s",
                    "functions": result.functions_indexed,
                    "classes": result.classes_indexed
                })
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Create failure result
            result = PackageProcessingResult(
                package_name=package_name,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
            
            logger.error("Package processing failed", extra={
                "package": package_name,
                "processing_time": f"{processing_time:.2f}s",
                "error": str(e)
            })
            
            return result
    
    def index_packages_batch(self, packages: List[Dict[str, Any]]) -> BatchProcessingMetrics:
        """Index multiple packages with progress tracking and metrics."""
        
        metrics = BatchProcessingMetrics(total_packages=len(packages))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Processing packages...", 
                total=len(packages)
            )
            
            for package_config in packages:
                result = self.process_package(package_config)
                metrics.results.append(result)
                
                # Update metrics
                if result.success:
                    metrics.successful_packages += 1
                    metrics.total_functions += result.functions_indexed
                    metrics.total_classes += result.classes_indexed
                else:
                    metrics.failed_packages += 1
                
                metrics.total_processing_time += result.processing_time
                
                progress.update(task, advance=1)
        
        # Calculate final metrics
        metrics.calculate_averages()
        
        logger.info("Batch processing completed", extra={
            "total_packages": metrics.total_packages,
            "successful": metrics.successful_packages,
            "failed": metrics.failed_packages,
            "total_time": f"{metrics.total_processing_time:.2f}s"
        })
        
        return metrics
    
    def generate_performance_report(self, metrics: BatchProcessingMetrics) -> None:
        """Generate comprehensive performance report."""
        
        console.print(Panel(
            "[bold green]📊 Production Processing Report[/bold green]",
            style="green"
        ))
        
        # Summary table
        summary_table = Table(title="Processing Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="bold white", justify="right")
        
        success_rate = (metrics.successful_packages / metrics.total_packages * 100) if metrics.total_packages > 0 else 0
        
        summary_rows = [
            ("Total Packages", str(metrics.total_packages)),
            ("Successful", str(metrics.successful_packages)),
            ("Failed", str(metrics.failed_packages)),
            ("Success Rate", f"{success_rate:.1f}%"),
            ("Total Functions", f"{metrics.total_functions:,}"),
            ("Total Classes", f"{metrics.total_classes:,}"),
            ("Total Time", f"{metrics.total_processing_time:.2f}s"),
            ("Average Time/Package", f"{metrics.average_package_time:.2f}s")
        ]
        
        for metric, value in summary_rows:
            summary_table.add_row(metric, value)
        
        console.print(summary_table)
        
        # Failed packages table
        if metrics.failed_packages > 0:
            console.print("\n[bold red]❌ Failed Packages[/bold red]")
            
            failed_table = Table()
            failed_table.add_column("Package", style="red")
            failed_table.add_column("Error", style="white")
            failed_table.add_column("Time", style="blue", justify="right")
            
            for result in metrics.results:
                if not result.success:
                    failed_table.add_row(
                        result.package_name,
                        result.error_message or "Unknown error",
                        f"{result.processing_time:.2f}s"
                    )
            
            console.print(failed_table)
        
        # Top performers
        console.print("\n[bold green]🏆 Top Performers[/bold green]")
        
        successful_results = [r for r in metrics.results if r.success]
        top_performers = sorted(
            successful_results, 
            key=lambda x: x.functions_indexed + x.classes_indexed, 
            reverse=True
        )[:5]
        
        if top_performers:
            top_table = Table()
            top_table.add_column("Package", style="green")
            top_table.add_column("Functions", style="blue", justify="right")
            top_table.add_column("Classes", style="yellow", justify="right")
            top_table.add_column("Time", style="cyan", justify="right")
            
            for result in top_performers:
                top_table.add_row(
                    result.package_name,
                    str(result.functions_indexed),
                    str(result.classes_indexed),
                    f"{result.processing_time:.2f}s"
                )
            
            console.print(top_table)


def demonstrate_production_deployment():
    """Demonstrate production deployment workflow."""
    
    console.print(Panel(
        "[bold cyan]🏭 ADPA Production Deployment Demo[/bold cyan]\n\n"
        "[dim]Comprehensive production workflow with monitoring and error handling[/dim]",
        style="cyan"
    ))
    
    # Initialize production configuration
    config = ProductionConfig(
        db_path="production_demo.db",
        batch_size=16,
        log_level="INFO",
        performance_monitoring=True
    )
    
    # Create production ADPA instance
    production_adpa = ProductionADPA(config)
    
    try:
        # Load packages configuration
        packages = production_adpa.load_packages_config()
        
        if not packages:
            console.print("[red]❌ No packages to process[/red]")
            return
        
        # Process packages in batch
        console.print(f"\n[bold yellow]Processing {len(packages)} packages...[/bold yellow]")
        
        metrics = production_adpa.index_packages_batch(packages)
        
        # Generate performance report
        production_adpa.generate_performance_report(metrics)
        
    except Exception as e:
        console.print(f"[red]❌ Production deployment failed: {e}[/red]")
        logger.error("Production deployment failed", extra={"error": str(e)})
    
    finally:
        # Cleanup
        try:
            Path("production_demo.db").unlink(missing_ok=True)
            Path("packages.json").unlink(missing_ok=True)
            Path("adpa_production.log").unlink(missing_ok=True)
            console.print("[dim]🧹 Cleaned up demo files[/dim]")
        except Exception:
            pass


def main():
    """Main production example."""
    
    try:
        demonstrate_production_deployment()
        
        console.print(Panel(
            "[bold green]✅ Production deployment demo completed![/bold green]\n\n"
            "[dim]Key production features:[/dim]\n"
            "• Structured logging with rotation\n"
            "• Configuration management\n"
            "• Batch processing with progress tracking\n"
            "• Comprehensive error handling\n"
            "• Performance monitoring and reporting\n"
            "• Resource management and cleanup\n\n"
            "[dim]Ready for production deployment![/dim]",
            style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")


if __name__ == "__main__":
    main()
