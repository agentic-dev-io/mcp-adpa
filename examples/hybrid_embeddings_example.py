#!/usr/bin/env python3
"""
🧠 ADPA Hybrid Embeddings Example - Modern Implementation

Demonstrates advanced graph neural network embeddings for better code understanding.
This example shows how to extend ADPA with PyTorch Geometric capabilities.

Note: This is a conceptual example. The actual implementation would require
additional modules for graph embeddings.
"""

import time
from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

from adpa import PackageIndexer

console = Console()


def check_graph_embeddings_available() -> bool:
    """Check if graph embeddings dependencies are available."""
    try:
        import torch
        import torch_geometric
        return True
    except ImportError:
        return False


def show_requirements_info():
    """Show information about graph embeddings requirements."""
    console.print(Panel(
        "[yellow]📦 Graph Embeddings Extension[/yellow]\n\n"
        "This example demonstrates how ADPA could be extended with\n"
        "PyTorch Geometric for advanced graph neural network embeddings.\n\n"
        "[dim]Dependencies for full graph support:[/dim]\n"
        "[cyan]pip install torch>=2.0.0 torch-geometric>=2.5.0[/cyan]\n\n"
        "[dim]Note: This is a conceptual demonstration of the capabilities.[/dim]",
        style="yellow",
        title="Graph Embeddings Info"
    ))


def compare_embedding_approaches():
    """Compare traditional vs potential hybrid embedding approaches."""
    
    console.print(Panel(
        "[bold cyan]🧠 Embedding Approaches Comparison[/bold cyan]\n\n"
        "[dim]Comparing current sentence transformers vs potential graph-enhanced embeddings[/dim]",
        style="cyan"
    ))
    
    # Test packages (small ones for demo)
    test_packages = ["json", "csv", "urllib.parse"]
    
    results = {}
    
    # Test current approach
    console.print(f"\n[bold yellow]Testing current sentence transformer approach...[/bold yellow]")
    
    current_results = []
    
    for package in track(test_packages, description="Processing with current approach..."):
        start_time = time.time()
        
        try:
            # Create indexer with current approach
            indexer = PackageIndexer(db_path=f"demo_current.db")
            
            # Index package
            package_results = indexer.index_package(package)
            index_time = time.time() - start_time
            
            # Test search performance
            search_start = time.time()
            search_results = indexer.search_similar("decode", threshold=0.1)
            search_time = time.time() - search_start
            
            # Count total results
            total_results = sum(len(items) for items in search_results.values())
            
            current_results.append({
                'package': package,
                'functions': package_results.get('functions', 0),
                'classes': package_results.get('classes', 0),
                'index_time': index_time,
                'search_time': search_time,
                'search_results': total_results
            })
            
        except Exception as e:
            console.print(f"[red]Error with {package}: {e}[/red]")
            current_results.append({
                'package': package,
                'functions': 0,
                'classes': 0,
                'index_time': 0,
                'search_time': 0,
                'search_results': 0,
                'error': str(e)
            })
    
    results['current'] = current_results
    
    # Simulate potential hybrid approach
    console.print(f"\n[bold yellow]Simulating potential hybrid approach...[/bold yellow]")
    
    hybrid_results = []
    
    for package in track(test_packages, description="Simulating hybrid approach..."):
        # Simulate additional processing time for graph features
        base_result = next((r for r in current_results if r['package'] == package), {})
        
        if 'error' not in base_result:
            # Simulate 20-30% longer indexing time for graph processing
            simulated_index_time = base_result.get('index_time', 0) * 1.25
            # Simulate slightly longer search time for enhanced features  
            simulated_search_time = base_result.get('search_time', 0) * 1.1
            # Simulate potentially better results
            simulated_results = int(base_result.get('search_results', 0) * 1.15)
            
            hybrid_results.append({
                'package': package,
                'functions': base_result.get('functions', 0),
                'classes': base_result.get('classes', 0),
                'index_time': simulated_index_time,
                'search_time': simulated_search_time,
                'search_results': simulated_results
            })
        else:
            hybrid_results.append(base_result)
    
    results['hybrid'] = hybrid_results
    
    # Display comparison
    console.print("\n[bold green]📊 Performance Comparison[/bold green]")
    
    comparison_table = Table()
    comparison_table.add_column("Package", style="cyan")
    comparison_table.add_column("Approach", style="yellow")
    comparison_table.add_column("Functions", style="white", justify="right")
    comparison_table.add_column("Index Time", style="blue", justify="right") 
    comparison_table.add_column("Search Time", style="green", justify="right")
    comparison_table.add_column("Results", style="magenta", justify="right")
    
    for package in test_packages:
        for approach_name, approach_key in [("Current", "current"), ("Hybrid (sim)", "hybrid")]:
            result = next((r for r in results[approach_key] if r['package'] == package), {})
            
            if 'error' in result:
                comparison_table.add_row(
                    package, approach_name, "Error", "Error", "Error", "Error"
                )
            else:
                comparison_table.add_row(
                    package,
                    approach_name,
                    str(result.get('functions', 0)),
                    f"{result.get('index_time', 0):.2f}s",
                    f"{result.get('search_time', 0)*1000:.0f}ms",
                    str(result.get('search_results', 0))
                )
    
    console.print(comparison_table)
    
    # Calculate and show averages
    show_performance_averages(results)
    
    # Clean up demo databases
    try:
        Path("demo_current.db").unlink(missing_ok=True)
    except Exception:
        pass


def show_performance_averages(results: Dict[str, List[Dict[str, Any]]]):
    """Display average performance metrics."""
    
    console.print("\n[bold green]📈 Average Performance[/bold green]")
    
    avg_table = Table()
    avg_table.add_column("Approach", style="cyan")
    avg_table.add_column("Avg Index Time", style="blue", justify="right")
    avg_table.add_column("Avg Search Time", style="green", justify="right")
    avg_table.add_column("Avg Results", style="magenta", justify="right")
    
    for approach_name, approach_key in [("Current", "current"), ("Hybrid (simulated)", "hybrid")]:
        valid_results = [r for r in results[approach_key] if 'error' not in r]
        
        if valid_results:
            avg_index = sum(r['index_time'] for r in valid_results) / len(valid_results)
            avg_search = sum(r['search_time'] for r in valid_results) / len(valid_results)
            avg_results = sum(r['search_results'] for r in valid_results) / len(valid_results)
            
            avg_table.add_row(
                approach_name,
                f"{avg_index:.2f}s",
                f"{avg_search*1000:.0f}ms", 
                f"{avg_results:.1f}"
            )
    
    console.print(avg_table)


def demonstrate_conceptual_features():
    """Demonstrate conceptual graph-specific features."""
    
    console.print("\n[bold yellow]🕸️ Conceptual Graph Features[/bold yellow]")
    
    # Show what graph embeddings could provide
    features_table = Table(title="🧠 Potential Graph Embedding Features")
    features_table.add_column("Feature", style="cyan")
    features_table.add_column("Benefit", style="green")
    features_table.add_column("Use Case", style="yellow")
    
    potential_features = [
        ("AST Structure", "Captures code syntax relationships", "Better structural similarity"),
        ("Call Graph Embedding", "Function interaction patterns", "Find related function groups"),
        ("Control Flow", "Execution path understanding", "Identify similar algorithms"),
        ("Data Flow", "Variable dependency tracking", "Find data transformation patterns"),
        ("Multi-scale Features", "Local + global code patterns", "Hierarchical code search")
    ]
    
    for feature, benefit, use_case in potential_features:
        features_table.add_row(feature, benefit, use_case)
    
    console.print(features_table)
    
    console.print("\n[bold green]🎯 Potential Applications:[/bold green]")
    applications = [
        "• More accurate semantic code search",
        "• Detection of similar algorithms across packages",
        "• Code pattern recommendation",
        "• Automated refactoring suggestions",
        "• Bug pattern detection"
    ]
    
    for app in applications:
        console.print(f"  {app}")


def main():
    """Main demonstration function."""
    
    # Check if graph embeddings are available
    has_graph_support = check_graph_embeddings_available()
    
    # Show requirements info
    show_requirements_info()
    
    if has_graph_support:
        console.print("\n[green]✅ Graph embedding dependencies are available![/green]")
    else:
        console.print("\n[yellow]📦 Graph embedding dependencies not installed[/yellow]")
    
    try:
        # Run performance comparison
        compare_embedding_approaches()
        
        # Demonstrate conceptual features
        demonstrate_conceptual_features()
        
        console.print(Panel(
            "[bold green]✅ Hybrid embeddings concept demo completed![/bold green]\n\n"
            "[dim]Key insights:[/dim]\n"
            "• Current approach provides solid baseline performance\n" 
            "• Graph embeddings could enhance structural understanding\n"
            "• Trade-offs exist between speed and advanced features\n"
            "• Integration would be valuable for complex code analysis\n\n"
            "[dim]Next steps for implementation:[/dim]\n"
            "• Design AST graph representation\n"
            "• Implement graph neural network layers\n"
            "• Create hybrid embedding fusion strategies\n"
            "• Benchmark on real-world code similarity tasks",
            style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")


if __name__ == "__main__":
    main()
