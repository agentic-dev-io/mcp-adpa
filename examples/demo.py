#!/usr/bin/env python3
"""
🚀 ADPA Package Demonstration - Modern Implementation

Demonstrates ADPA features with DuckDB and modern ML models.
Focus on clean code patterns and robust error handling.
"""

from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

from adpa import CodeContextAnalyzer, PackageIndexer

console = Console()


def demonstrate_package_indexing():
    """Demonstrates package indexing with modern patterns."""
    
    console.print(Panel(
        "[bold cyan]📦 Package Indexer with Modern Stack[/bold cyan]\n\n"
        "[dim]DuckDB • Modern Embeddings • Clean Architecture[/dim]",
        style="cyan"
    ))
    
    # Initialize components
    indexer = PackageIndexer(db_path="demo_packages.db")
    context_analyzer = CodeContextAnalyzer(indexer)
    
    # Test packages (focused on async/typing features)
    packages = ["pathlib", "asyncio", "dataclasses", "typing"]
    
    console.print("\n[bold yellow]📚 Indexing packages...[/bold yellow]")
    
    for package in track(packages, description="Processing packages..."):
        try:
            indexer.index_package(package)
            console.print(f"✅ Indexed: [cyan]{package}[/cyan]")
        except Exception as e:
            console.print(f"❌ Failed to index {package}: {e}")
    
    return indexer, context_analyzer


def show_package_statistics(indexer: PackageIndexer):
    """Display comprehensive package statistics."""
    
    console.print("\n[bold yellow]📊 Package Statistics[/bold yellow]")
    
    try:
        stats = indexer.get_package_stats()
        totals = stats.get('totals', {})
        
        # Create statistics table
        stats_table = Table(title="Database Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="bold white", justify="right")
        
        key_metrics = [
            ("packages", "Total packages"),
            ("functions", "Total functions"),
            ("classes", "Total classes"),
            ("dataclasses", "Dataclasses"),
            ("avg_complexity", "Average complexity")
        ]
        
        for key, label in key_metrics:
            value = totals.get(key, 0)
            if key == "avg_complexity":
                display_value = f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"
            else:
                display_value = f"{value:,}" if isinstance(value, int) else str(value)
            
            stats_table.add_row(label, display_value)
        
        console.print(stats_table)
        
    except Exception as e:
        console.print(f"[red]❌ Stats error: {e}[/red]")


def demonstrate_similarity_search(indexer: PackageIndexer):
    """Demonstrate semantic similarity search."""
    
    console.print("\n[bold yellow]🔍 Semantic Similarity Search[/bold yellow]")
    
    queries = [
        "async file operations",
        "dataclass with validation", 
        "type hints for functions",
        "context manager for resources",
    ]
    
    for query in queries:
        console.print(f"\nQuery: '[cyan]{query}[/cyan]'")
        
        try:
            results = indexer.search_similar(
                query, 
                limit=3, 
                similarity_threshold=0.3
            )
            
            # Display function results
            functions = results.get("functions", [])[:2]
            for func in functions:
                package = func.get('package', 'Unknown')
                name = func.get('name', 'Unknown')
                is_async = func.get('is_async', False)
                similarity = func.get('similarity', 0)
                decorators = func.get('decorators', [])
                
                console.print(f"  📄 Function: [bold]{package}.{name}[/bold]")
                console.print(f"    Async: {is_async}, Similarity: [green]{similarity:.3f}[/green]")
                console.print(f"    Decorators: {decorators}")
            
            # Display class results
            classes = results.get("classes", [])[:1]
            for cls in classes:
                package = cls.get('package', 'Unknown')
                name = cls.get('name', 'Unknown')
                is_dataclass = cls.get('is_dataclass', False)
                similarity = cls.get('similarity', 0)
                
                console.print(f"  📝 Class: [bold]{package}.{name}[/bold]")
                console.print(f"    Dataclass: {is_dataclass}, Similarity: [green]{similarity:.3f}[/green]")
                
        except Exception as e:
            console.print(f"[red]❌ Search error: {e}[/red]")


def demonstrate_code_analysis(context_analyzer: CodeContextAnalyzer):
    """Demonstrate advanced code pattern analysis."""
    
    console.print("\n[bold yellow]🧠 Code Pattern Analysis[/bold yellow]")
    
    # Modern Python code sample
    sample_code = '''
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class User:
    name: str
    age: int
    email: Optional[str] = None

async def fetch_users() -> List[User]:
    users = [User("Alice", 30, "alice@example.com") for i in range(10)]
    return users

async def main():
    async with asyncio.timeout(5.0):
        users = await fetch_users()
        print(f"Fetched {len(users)} users")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    try:
        patterns = context_analyzer.analyze_patterns(sample_code)
        
        # Create analysis table
        analysis_table = Table(title="Pattern Analysis Results")
        analysis_table.add_column("Category", style="cyan")
        analysis_table.add_column("Metric", style="yellow")
        analysis_table.add_column("Value", style="white", justify="right")
        
        # Extract and display patterns
        analyses = [
            ("Async Usage", "async_functions", patterns.get('async_usage', {}).get('async_functions', 0)),
            ("Type Hints", "annotation_coverage", patterns.get('type_hints', {}).get('annotation_coverage', 0)),
            ("Decorators", "unique_decorators", len(patterns.get('decorators', {}).get('unique_decorators', []))),
            ("Future Imports", "count", len(patterns.get('future_imports', [])))
        ]
        
        for category, metric, value in analyses:
            if metric == "annotation_coverage":
                display_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
            else:
                display_value = str(value)
            
            analysis_table.add_row(category, metric.replace('_', ' ').title(), display_value)
        
        console.print(analysis_table)
        
        # Show specific findings
        console.print("\n[bold green]Key Findings:[/bold green]")
        async_usage = patterns.get('async_usage', {})
        console.print(f"• Async functions detected: {async_usage.get('async_functions', 0)}")
        
        type_hints = patterns.get('type_hints', {})
        coverage = type_hints.get('annotation_coverage', 0)
        console.print(f"• Type annotation coverage: {coverage:.1%}")
        
        decorators = patterns.get('decorators', {})
        unique_decorators = decorators.get('unique_decorators', [])
        console.print(f"• Decorators used: {', '.join(unique_decorators) if unique_decorators else 'None'}")
        
        future_imports = patterns.get('future_imports', [])
        console.print(f"• Future imports: {', '.join(future_imports) if future_imports else 'None'}")
        
    except Exception as e:
        console.print(f"[red]❌ Analysis error: {e}[/red]")


def main():
    """Main demonstration function."""
    
    try:
        # Index packages
        indexer, context_analyzer = demonstrate_package_indexing()
        
        # Run demonstrations
        show_package_statistics(indexer)
        demonstrate_similarity_search(indexer)
        demonstrate_code_analysis(context_analyzer)
        
        # Clean up
        try:
            from pathlib import Path
            Path("demo_packages.db").unlink(missing_ok=True)
        except Exception:
            pass
        
        console.print(Panel(
            "[bold green]✅ Demonstration completed successfully![/bold green]\n\n"
            "[dim]Next steps:[/dim]\n"
            "• Explore semantic search capabilities\n"
            "• Analyze your own codebase patterns\n"
            "• Integrate with development workflow",
            style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")


if __name__ == "__main__":
    main()
