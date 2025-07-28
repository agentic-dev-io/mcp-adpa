#!/usr/bin/env python3
"""
🚀 ADPA Basic Usage Examples - Modern Implementation

This script demonstrates the core functionality of ADPA for package analysis
and semantic code search using modern Python patterns.
"""

from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from adpa import PackageIndexer

console = Console()


def demonstrate_quick_indexing():
    """Demonstrate quick package indexing with modern patterns."""
    
    console.print("[bold yellow]📚 Example 1: Package Indexing[/bold yellow]")
    
    indexer = PackageIndexer(db_path="demo.db")
    
    try:
        console.print("Indexing the 'json' package...")
        results = indexer.index_package("json")
        
        # Modern table display
        results_table = Table(title="📊 Indexing Results")
        results_table.add_column("Component", style="cyan")
        results_table.add_column("Count", style="bold green", justify="right")
        
        for component, count in results.items():
            results_table.add_row(component.title(), str(count))
        
        console.print(results_table)
        console.print()
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error indexing package: {e}[/red]\n")
        return False


def demonstrate_semantic_search(indexer: PackageIndexer):
    """Demonstrate semantic search capabilities."""
    
    console.print("[bold yellow]🔍 Example 2: Semantic Search[/bold yellow]")
    
    search_queries = [
        "decode json",
        "parse string", 
        "load data"
    ]
    
    for query in search_queries:
        console.print(f"Searching for: [cyan]'{query}'[/cyan]")
        
        try:
            results = indexer.search_similar(
                query, 
                threshold=0.1
            )
            
            # Create tree view of results
            tree = Tree(f"🔍 Results for '{query}'")
            
            for category, items in results.items():
                if items:
                    category_branch = tree.add(
                        f"[yellow]{category.title()}[/yellow] ({len(items)} found)"
                    )
                    
                    # Show top 3 results
                    for item in items[:3]:
                        similarity = item.get('similarity', 0)
                        name = item.get('name', 'Unknown')
                        package = item.get('package', 'Unknown')
                        
                        similarity_color = "green" if similarity > 0.5 else "yellow"
                        item_text = (
                            f"[bold]{name}[/bold] "
                            f"[{similarity_color}]({similarity:.2f})[/{similarity_color}] "
                            f"[dim]- {package}[/dim]"
                        )
                        category_branch.add(item_text)
            
            console.print(tree)
            console.print()
            
        except Exception as e:
            console.print(f"[red]❌ Search error: {e}[/red]\n")


def demonstrate_advanced_features(indexer: PackageIndexer):
    """Demonstrate advanced indexer features."""
    
    console.print("[bold yellow]⚙️ Example 3: Advanced Features[/bold yellow]")
    
    try:
        # Get comprehensive statistics
        console.print("Getting package statistics...")
        stats = indexer.get_package_stats()
        
        # Display statistics
        stats_table = Table(title="📊 Database Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="bold white", justify="right")
        
        totals = stats.get('totals', {})
        for metric, value in totals.items():
            if isinstance(value, (int, float)):
                display_value = f"{value:,}" if isinstance(value, int) else f"{value:.2f}"
            else:
                display_value = str(value) if value is not None else "N/A"
            
            stats_table.add_row(metric.replace('_', ' ').title(), display_value)
        
        console.print(stats_table)
        console.print()
        
        # Show model information if available
        console.print("Getting model information...")
        
        try:
            model_info = indexer.scanner.get_model_info()
            
            model_table = Table(title="🧠 Model Information")
            model_table.add_column("Property", style="cyan")
            model_table.add_column("Value", style="white")
            
            for key, value in model_info.items():
                if not key.startswith('_'):  # Skip private attributes
                    model_table.add_row(
                        key.replace('_', ' ').title(), 
                        str(value)
                    )
            
            console.print(model_table)
            console.print()
            
        except Exception as e:
            console.print(f"[dim]Model info not available: {e}[/dim]\n")
        
    except Exception as e:
        console.print(f"[red]❌ Advanced features error: {e}[/red]\n")


def demonstrate_call_graph_analysis(indexer: PackageIndexer):
    """Demonstrate call graph analysis."""
    
    console.print("[bold yellow]🕸️ Example 4: Call Graph Analysis[/bold yellow]")
    
    try:
        # Analyze call relationships in the json package
        call_query = """
            SELECT source_function, target_function, COUNT(*) as call_count
            FROM call_graph 
            WHERE package = 'json'
            GROUP BY source_function, target_function
            ORDER BY call_count DESC
            LIMIT 5
        """
        
        console.print("Analyzing function call relationships...")
        call_results = indexer.conn.execute(call_query).fetchall()
        
        if call_results:
            call_table = Table(title="🕸️ Function Call Relationships")
            call_table.add_column("Source Function", style="cyan")
            call_table.add_column("Target Function", style="yellow")
            call_table.add_column("Call Count", style="bold green", justify="right")
            
            for source, target, count in call_results:
                call_table.add_row(source, target, str(count))
            
            console.print(call_table)
        else:
            console.print("[dim]No call graph data available for json package[/dim]")
        
        console.print()
        
    except Exception as e:
        console.print(f"[red]❌ Call graph error: {e}[/red]\n")


def main():
    """Main demonstration function."""
    
    console.print(Panel(
        "[bold cyan]🚀 ADPA Demo - Package Analysis & Semantic Search[/bold cyan]\n\n"
        "[dim]This demo shows how to use ADPA for intelligent code discovery[/dim]",
        style="cyan"
    ))
    
    # Run demonstrations
    if not demonstrate_quick_indexing():
        return
    
    # Create indexer for subsequent demos
    indexer = PackageIndexer(db_path="demo.db")
    
    # Run all demonstrations
    demonstrate_semantic_search(indexer)
    demonstrate_advanced_features(indexer)
    demonstrate_call_graph_analysis(indexer)
    
    # Clean up demo database
    try:
        Path("demo.db").unlink(missing_ok=True)
        console.print("[dim]🧹 Cleaned up demo database[/dim]")
    except Exception:
        pass
    
    # Final message
    console.print(Panel(
        "[bold green]✅ Demo completed successfully![/bold green]\n\n"
        "[dim]Next steps:[/dim]\n"
        "• Try: [cyan]adpa index requests[/cyan]\n"
        "• Try: [cyan]adpa search \"http client\"[/cyan]\n" 
        "• Try: [cyan]adpa stats --detailed[/cyan]\n\n"
        "[dim]For more examples, see the documentation at:[/dim]\n"
        "📚 https://github.com/your-org/adpa",
        style="green"
    ))


if __name__ == "__main__":
    main()
