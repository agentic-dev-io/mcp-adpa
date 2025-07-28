#!/usr/bin/env python3
"""
🚀 Quick Demo of ADPA - Modern Implementation

Demonstrates ADPA capabilities with modern Python patterns and
clean error handling.
"""

from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from adpa import PackageIndexer

console = Console()


def quick_demo():
    """Quick demonstration of key features with modern patterns."""
    
    console.print(Panel(
        "[bold cyan]🚀 ADPA - Quick Demo[/bold cyan]\n\n"
        "[dim]Modern package analysis with semantic search[/dim]",
        style="cyan"
    ))
    
    # Initialize indexer with correct parameters
    indexer = PackageIndexer(db_path="demo.db")
    
    # Index a simple package
    console.print("\n📦 Indexing 'json' package...")
    
    try:
        results = indexer.index_package("json")
        
        # Display results in modern table format
        results_table = Table(title="📊 Indexing Results")
        results_table.add_column("Component", style="cyan")
        results_table.add_column("Count", style="bold green", justify="right")
        
        for component, count in results.items():
            results_table.add_row(component.title(), str(count))
        
        console.print(results_table)
        
    except Exception as e:
        console.print(f"[red]❌ Indexing failed: {e}[/red]")
        return
    
    # Show package statistics
    console.print("\n📊 Package Statistics:")
    
    try:
        stats = indexer.get_package_stats()
        totals = stats.get('totals', {})
        
        stats_table = Table()
        stats_table.add_column("Metric", style="yellow")
        stats_table.add_column("Value", style="white", justify="right")
        
        metrics = [
            ("packages", "📁 Packages"),
            ("functions", "🔧 Functions"),
            ("classes", "📝 Classes"),
            ("avg_complexity", "📈 Avg Complexity")
        ]
        
        for key, label in metrics:
            value = totals.get(key, 0)
            if key == "avg_complexity":
                display_value = f"{value:.1f}" if isinstance(value, (int, float)) else "N/A"
            else:
                display_value = str(value)
            
            stats_table.add_row(label, display_value)
        
        console.print(stats_table)
        
    except Exception as e:
        console.print(f"[red]❌ Stats failed: {e}[/red]")
    
    # Semantic search examples
    console.print("\n🔍 Semantic Search Examples:")
    
    search_queries = [
        ("loads", "Finding JSON loading functions"),
        ("decode json", "Finding JSON decoding functions"),
        ("parse", "Finding parsing functions")
    ]
    
    for query, description in search_queries:
        console.print(f"\n   🔎 {description}")
        console.print(f"   Query: '[cyan]{query}[/cyan]'")
        
        try:
            results = indexer.search_similar(
                query, 
                limit=2, 
                similarity_threshold=0.2
            )
            
            functions = results.get('functions', [])[:2]
            
            if functions:
                for func in functions:
                    name = func.get('name', 'Unknown')
                    similarity = func.get('similarity', 0)
                    signature = func.get('signature', '')
                    
                    console.print(f"   ✨ {name} (similarity: [green]{similarity:.3f}[/green])")
                    
                    if signature:
                        console.print(f"      [dim]{signature}[/dim]")
            else:
                console.print("   [dim]No results found[/dim]")
                
        except Exception as e:
            console.print(f"   [red]❌ Search error: {e}[/red]")
    
    # Cleanup
    try:
        Path("demo.db").unlink(missing_ok=True)
    except Exception:
        pass
    
    console.print(Panel(
        "[bold green]🎉 Demo completed successfully![/bold green]\n\n"
        "[dim]Next steps:[/dim]\n"
        "• [cyan]uv run adpa search 'decode json' --threshold 0.2[/cyan]\n"
        "• [cyan]uv run adpa call-graph --source 'loads'[/cyan]\n"
        "• [cyan]uv run adpa stats[/cyan]",
        style="green"
    ))


if __name__ == "__main__":
    quick_demo()
