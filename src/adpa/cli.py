"""
🚀 ADPA CLI - Modern Interactive Python Package Analyzer

Beautiful CLI inspired by Claude Code with interactive features,
rich output, and comprehensive examples.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import wraps
import subprocess

import rich_click as click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.traceback import install as install_rich_traceback
from rich.prompt import Prompt, Confirm
from rich.tree import Tree
from rich.columns import Columns
from rich.text import Text

from .indexer import PackageIndexer

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 RICH-CLICK CONFIGURATION - Claude Code Style
# ═══════════════════════════════════════════════════════════════════════════════

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.APPEND_METAVARS_HELP = True

# Modern styling inspired by Claude Code
click.rich_click.STYLE_OPTION = "bold cyan"
click.rich_click.STYLE_ARGUMENT = "bold yellow" 
click.rich_click.STYLE_COMMAND = "bold green"
click.rich_click.STYLE_SWITCH = "bold magenta"
click.rich_click.STYLE_METAVAR = "bold blue"
click.rich_click.STYLE_HEADER_TEXT = "bold white"
click.rich_click.STYLE_HELPTEXT_FIRST_LINE = "bold"
click.rich_click.STYLE_HELPTEXT = "dim"
click.rich_click.STYLE_OPTION_DEFAULT = "dim cyan"

# Layout
click.rich_click.MAX_WIDTH = 120
click.rich_click.COLOR_SYSTEM = "auto"
click.rich_click.FORCE_TERMINAL = True

# Install rich traceback for beautiful errors
install_rich_traceback(show_locals=True, width=120, extra_lines=3, suppress=[click])

# Rich console
console = Console(width=120, color_system="auto", force_terminal=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure loguru logging with rich integration."""
    logger.remove()
    
    if debug:
        level = "DEBUG"
        format_str = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    elif verbose:
        level = "INFO"  
        format_str = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    else:
        level = "WARNING"
        format_str = "<level>{level}</level> | <level>{message}</level>"
    
    logger.add(
        sys.stderr,
        level=level,
        format=format_str,
        colorize=True,
        backtrace=True,
        enqueue=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 🛡️ UTILITIES & ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def handle_errors(func):
    """🛡️ Decorator for consistent error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            console.print("\n[yellow]⏹️  Operation cancelled by user[/yellow]")
            sys.exit(130)
        except Exception as e:
            console.print(Panel(
                f"[red]💥 Error: {e}[/red]",
                style="red",
                title="❌ Error",
                border_style="red"
            ))
            logger.exception(f"Error in {func.__name__}")
            sys.exit(1)
    return wrapper

def create_header(title: str, subtitle: str = "") -> Panel:
    """🎨 Create beautiful header panel."""
    content = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"
    
    return Panel(
        content, 
        style="cyan", 
        border_style="bright_cyan",
        padding=(1, 2),
        title="🚀 ADPA",
        title_align="left"
    )

def get_indexer(db_path: Path, verbose: bool = False) -> PackageIndexer:
    """🔧 Create indexer with error handling."""
    try:
        return PackageIndexer(db_path=str(db_path))
    except Exception as e:
        if verbose:
            logger.error(f"Failed to initialize indexer: {e}")
        console.print(Panel(
            f"[red]Failed to initialize ADPA indexer: {e}[/red]\n\n"
            f"[dim]Check your database path: {db_path}[/dim]",
            style="red",
            title="❌ Initialization Error",
            border_style="red"
        ))
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 MAIN CLI GROUP
# ═══════════════════════════════════════════════════════════════════════════════

@click.group(invoke_without_command=True)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output with detailed logging"
)
@click.option(
    "--debug", "-d", 
    is_flag=True,
    help="Enable debug mode with full logging and diagnostics"
)
@click.option(
    "--db-path",
    type=click.Path(),
    default="package_index.db",
    help="Path to DuckDB database file",
    show_default=True
)
@click.pass_context
def cli(ctx, verbose: bool, debug: bool, db_path: str):
    """🚀 **ADPA - AST DuckDB Package Analyzer**

    Modern Python package analysis with semantic search capabilities.
    """
    # Setup logging
    setup_logging(verbose, debug)
    
    # Store context
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug
    ctx.obj["db_path"] = Path(db_path)
    
    # If no command provided, show interactive menu
    if ctx.invoked_subcommand is None:
        show_interactive_menu(ctx)

def show_interactive_menu(ctx):
    """🎮 Show interactive menu like Claude Code."""
    
    # Header
    console.print(Panel(
        "[bold cyan]🚀 ADPA - AST DuckDB Package Analyzer[/bold cyan]\n\n"
        "[dim]Interactive Python package analysis with semantic search[/dim]\n\n"
        "Features:\n"
        "• 🔍 Semantic code search using embeddings\n"
        "• 📊 Advanced analytics with DuckDB\n"
        "• 🎯 AST-based code analysis\n"
        "• ⚡ High-performance indexing",
        style="cyan",
        border_style="bright_cyan",
        title="Welcome",
        title_align="left",
        padding=(1, 2)
    ))
    
    while True:
        console.print("\n" + "─" * 120)
        
        # Main menu options
        menu_options = [
            ("1", "📦 Index Package", "index", "Analyze and index a Python package"),
            ("2", "🔍 Search Code", "search", "Semantic search through indexed code"),
            ("3", "📊 Show Statistics", "stats", "Display database statistics"),
            ("4", "🕸️ Call Graph", "call-graph", "Analyze function call relationships"),
            ("5", "📚 Run Examples", "examples", "Interactive examples and demos"),
            ("6", "🗑️ Clean Database", "clean", "Clear all indexed data"),
            ("7", "❓ Help", "help", "Show detailed help information"),
            ("q", "🚪 Quit", "quit", "Exit ADPA")
        ]
        
        # Create menu table
        table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            width=120
        )
        table.add_column("Key", style="bold cyan", width=5)
        table.add_column("Action", style="bold white", width=20)
        table.add_column("Command", style="dim cyan", width=15)
        table.add_column("Description", style="dim", width=80)
        
        for key, action, command, description in menu_options:
            table.add_row(key, action, command, description)
        
        console.print(table)
        
        # Get user choice
        choice = Prompt.ask(
            "\n[bold cyan]Choose an option[/bold cyan]",
            choices=[opt[0] for opt in menu_options],
            default="1"
        ).lower()
        
        if choice == "q":
            console.print("\n[green]👋 Thanks for using ADPA![/green]")
            break
        elif choice == "1":
            interactive_index(ctx)
        elif choice == "2":
            interactive_search(ctx)
        elif choice == "3":
            interactive_stats(ctx)
        elif choice == "4":
            interactive_call_graph(ctx)
        elif choice == "5":
            interactive_examples(ctx)
        elif choice == "6":
            interactive_clean(ctx)
        elif choice == "7":
            show_help()

def interactive_index(ctx):
    """📦 Interactive package indexing."""
    console.print(create_header("📦 Interactive Package Indexing"))
    
    # Get package name
    package_name = Prompt.ask(
        "[bold cyan]Enter package name to index[/bold cyan]",
        default="requests"
    )
    
    # Options
    force = Confirm.ask("Force re-indexing if package exists?", default=False)
    
    # Run indexing
    ctx.invoke(index, package_name=package_name, clear=False, force=force)

def interactive_search(ctx):
    """🔍 Interactive semantic search."""
    console.print(create_header("🔍 Interactive Semantic Search"))
    
    # Get search query
    query = Prompt.ask(
        "[bold cyan]Enter search query[/bold cyan]",
        default="http client"
    )
    
    # Advanced options
    advanced = Confirm.ask("Configure advanced options?", default=False)
    
    if advanced:
        limit = int(Prompt.ask("Number of results", default="10"))
        threshold = float(Prompt.ask("Similarity threshold (0.0-1.0)", default="0.3"))
        search_type = Prompt.ask(
            "Search type",
            choices=["all", "functions", "classes", "modules"],
            default="all"
        )
    else:
        limit = 10
        threshold = 0.3
        search_type = "all"
    
    # Run search
    ctx.invoke(search, query=query, limit=limit, threshold=threshold, 
               package=None, type=search_type)

def interactive_stats(ctx):
    """📊 Interactive statistics viewer."""
    console.print(create_header("📊 Interactive Statistics"))
    
    # Options
    detailed = Confirm.ask("Show detailed statistics?", default=True)
    package = Prompt.ask(
        "Specific package (or press Enter for all)", 
        default="", 
        show_default=False
    )
    
    package = package if package else None
    
    # Run stats
    ctx.invoke(stats, package=package, detailed=detailed)

def interactive_call_graph(ctx):
    """🕸️ Interactive call graph analysis."""
    console.print(create_header("🕸️ Interactive Call Graph Analysis"))
    
    # Options
    source = Prompt.ask(
        "Source function pattern (or press Enter for all)",
        default="",
        show_default=False
    )
    package = Prompt.ask(
        "Specific package (or press Enter for all)",
        default="",
        show_default=False
    )
    limit = int(Prompt.ask("Maximum results", default="20"))
    
    source = source if source else None
    package = package if package else None
    
    # Run call graph
    ctx.invoke(call_graph, source=source, package=package, limit=limit)

def interactive_clean(ctx):
    """🗑️ Interactive database cleaning."""
    console.print(create_header("🗑️ Interactive Database Cleaning"))
    
    console.print(Panel(
        "[yellow]⚠️ This will permanently delete all indexed data![/yellow]\n\n"
        "[dim]This action cannot be undone. You will need to re-index\n"
        "all packages after cleaning the database.[/dim]",
        style="yellow",
        title="⚠️ Warning",
        border_style="yellow"
    ))
    
    if Confirm.ask("[bold red]Are you sure you want to continue?[/bold red]", default=False):
        ctx.invoke(clean, force=True)
    else:
        console.print("[green]Operation cancelled.[/green]")

def interactive_examples(ctx):
    """📚 Interactive examples menu."""
    console.print(create_header("📚 Interactive Examples & Demos"))
    
    examples = [
        {
            "name": "🚀 Quick Demo",
            "file": "quick_demo.py",
            "description": "Basic ADPA functionality demonstration"
        },
        {
            "name": "📊 Basic Usage",
            "file": "basic_usage.py", 
            "description": "Comprehensive usage examples with Rich output"
        },
        {
            "name": "🎯 Demo Features",
            "file": "demo.py",
            "description": "Package indexing and semantic search demo"
        },
        {
            "name": "🧠 Hybrid Embeddings",
            "file": "hybrid_embeddings_example.py",
            "description": "Advanced graph neural network embeddings (requires PyTorch Geometric)"
        },
        {
            "name": "🏭 Production Example",
            "file": "production_example.py",
            "description": "Production deployment with monitoring and batch processing"
        }
    ]
    
    # Show examples table
    table = Table(
        title="📚 Available Examples",
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        width=120
    )
    
    table.add_column("Key", style="bold cyan", width=5)
    table.add_column("Example", style="bold white", width=30)
    table.add_column("File", style="dim cyan", width=25)
    table.add_column("Description", style="dim", width=60)
    
    for i, example in enumerate(examples, 1):
        table.add_row(
            str(i),
            example["name"],
            example["file"],
            example["description"]
        )
    
    console.print(table)
    
    # Get choice
    choice = Prompt.ask(
        "\n[bold cyan]Choose an example to run[/bold cyan]",
        choices=[str(i) for i in range(1, len(examples) + 1)] + ["b"],
        default="1"
    )
    
    if choice == "b":
        return
    
    example_idx = int(choice) - 1
    example = examples[example_idx]
    
    # Confirm and run
    console.print(f"\n[bold green]Running: {example['name']}[/bold green]")
    console.print(f"[dim]{example['description']}[/dim]")
    
    if Confirm.ask("\nProceed?", default=True):
        run_example(example["file"])

def run_example(filename: str):
    """🏃 Run an example file."""
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    example_path = examples_dir / filename
    
    if not example_path.exists():
        console.print(f"[red]❌ Example file not found: {example_path}[/red]")
        return
    
    console.print(Panel(
        f"[bold cyan]Running: {filename}[/bold cyan]\n\n"
        f"[dim]Path: {example_path}[/dim]",
        style="cyan",
        title="🏃 Executing Example",
        border_style="cyan"
    ))
    
    try:
        # Run the example
        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=False,
            text=True,
            cwd=examples_dir.parent
        )
        
        if result.returncode == 0:
            console.print(Panel(
                "[green]✅ Example completed successfully![/green]",
                style="green",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[red]❌ Example failed with exit code {result.returncode}[/red]",
                style="red",
                border_style="red"
            ))
            
    except Exception as e:
        console.print(Panel(
            f"[red]❌ Failed to run example: {e}[/red]",
            style="red",
            border_style="red"
        ))
    
    input("\nPress Enter to continue...")

def show_help():
    """❓ Show detailed help information."""
    console.print(create_header("❓ ADPA Help & Documentation"))
    
    help_sections = [
        {
            "title": "🚀 Getting Started",
            "content": [
                "1. Index a package: `adpa index requests`",
                "2. Search for code: `adpa search \"http client\"`",
                "3. View statistics: `adpa stats`",
                "4. Analyze calls: `adpa call-graph -s main`"
            ]
        },
        {
            "title": "📦 Indexing Packages",
            "content": [
                "• `adpa index <package>` - Index a Python package",
                "• `adpa index <package> --force` - Force re-indexing", 
                "• `adpa index <package> --clear` - Clear before indexing",
                "• Supports any importable Python package"
            ]
        },
        {
            "title": "🔍 Semantic Search",
            "content": [
                "• `adpa search \"query\"` - Search all indexed code",
                "• `adpa search \"query\" -n 5` - Limit to 5 results",
                "• `adpa search \"query\" -t 0.8` - High similarity only",
                "• `adpa search \"query\" -p package` - Search specific package",
                "• `adpa search \"query\" -T functions` - Search only functions"
            ]
        },
        {
            "title": "📊 Analytics",
            "content": [
                "• `adpa stats` - Overall statistics",
                "• `adpa stats -p package` - Package-specific stats",
                "• `adpa stats --detailed` - Detailed analytics",
                "• `adpa call-graph` - Function call relationships"
            ]
        },
        {
            "title": "🛠️ Advanced Features",
            "content": [
                "• DuckDB backend for analytics",
                "• Sentence transformer embeddings",
                "• AST-based code analysis",
                "• Call graph generation",
                "• Hybrid embeddings (with PyTorch Geometric)"
            ]
        }
    ]
    
    for section in help_sections:
        panel_content = "\n".join(f"  {item}" for item in section["content"])
        console.print(Panel(
            panel_content,
            title=section["title"],
            border_style="cyan",
            padding=(1, 2)
        ))
    
    console.print(Panel(
        "[bold cyan]📚 For more examples, use: `adpa` → Option 5[/bold cyan]\n\n"
        "[dim]Documentation: https://github.com/your-org/adpa[/dim]",
        style="green",
        title="💡 Next Steps",
        border_style="green"
    ))
    
    input("\nPress Enter to continue...")

# ═══════════════════════════════════════════════════════════════════════════════
# 📦 INDEX COMMAND
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument("package_name", required=True)
@click.option("--clear", "-c", is_flag=True, help="🗑️ Clear existing index before indexing")
@click.option("--force", "-f", is_flag=True, help="⚡ Force re-indexing even if package exists")
@click.pass_context
@handle_errors
def index(ctx, package_name: str, clear: bool, force: bool):
    """📦 Index a Python package for semantic search."""
    
    console.print(create_header(
        f"📦 Indexing Package: {package_name}",
        "Analyzing AST and building search index..."
    ))
    
    indexer = get_indexer(ctx.obj["db_path"], ctx.obj["verbose"])
    
    if clear:
        console.print("[yellow]🗑️  Clearing existing index...[/yellow]")
        try:
            indexer.clear_database()
            logger.info("Database cleared successfully")
        except Exception as e:
            console.print(f"[red]❌ Failed to clear database: {e}[/red]")
            return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(f"Indexing {package_name}...", total=None)
        
        try:
            stats = indexer.index_package(package_name, force_reindex=force)
            
            console.print(Panel(
                f"[green]✅ Successfully indexed {package_name}![/green]\n\n"
                f"📊 Statistics:\n"
                f"• Functions: {stats.get('functions', 0)}\n"
                f"• Classes: {stats.get('classes', 0)}\n"
                f"• Modules: {stats.get('modules', 0)}\n"
                f"• Call graph edges: {stats.get('calls', 0)}",
                style="green",
                title="✨ Indexing Complete",
                border_style="green"
            ))
            
            logger.info(f"Successfully indexed {package_name} with {stats}")
            
        except ImportError as e:
            console.print(Panel(
                f"[red]❌ Package '{package_name}' not found or not importable[/red]\n\n"
                f"[dim]Error: {e}[/dim]\n\n"
                f"[yellow]Try installing the package first:[/yellow]\n"
                f"[cyan]pip install {package_name}[/cyan]",
                style="red",
                title="❌ Import Error",
                border_style="red"
            ))
            logger.error(f"Failed to import package {package_name}: {e}")
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Failed to index {package_name}[/red]\n\n"
                f"[dim]Error: {e}[/dim]",
                style="red",
                title="❌ Indexing Error",
                border_style="red"
            ))
            logger.error(f"Failed to index {package_name}: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 🔍 SEARCH COMMAND
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument("query", required=True)
@click.option("--limit", "-n", type=int, default=10, help="Number of results", show_default=True)
@click.option("--threshold", "-t", type=float, default=0.3, help="Similarity threshold", show_default=True)
@click.option("--package", "-p", help="Limit search to specific package")
@click.option("--type", "-T", type=click.Choice(['functions', 'classes', 'modules', 'all']), 
              default='all', help="Type of items to search", show_default=True)
@click.pass_context
@handle_errors  
def search(ctx, query: str, limit: int, threshold: float, package: Optional[str], type: str):
    """🔍 Semantic search through indexed code."""
    
    if not 0.0 <= threshold <= 1.0:
        console.print("[red]❌ Threshold must be between 0.0 and 1.0[/red]")
        return
    
    subtitle_parts = [f"Limit: {limit}", f"Threshold: {threshold}"]
    if package:
        subtitle_parts.append(f"Package: {package}")
    if type != 'all':
        subtitle_parts.append(f"Type: {type}")
        
    console.print(create_header(
        f"🔍 Searching: {query}",
        " | ".join(subtitle_parts)
    ))
    
    indexer = get_indexer(ctx.obj["db_path"], ctx.obj["verbose"])
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Searching..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Searching...", total=None)
        
        try:
            results = indexer.search_similar(
                query=query,
                limit=limit,
                similarity_threshold=threshold
            )
            logger.info(f"Search for '{query}' returned {sum(len(items) for items in results.values())} results")
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Search failed[/red]\n\n"
                f"[dim]Error: {e}[/dim]",
                style="red",
                title="❌ Search Error",
                border_style="red"
            ))
            logger.error(f"Search failed: {e}")
            return
    
    # Filter results
    if package:
        for category in results:
            results[category] = [
                item for item in results[category] 
                if item.get("package", "").lower() == package.lower()
            ]
    
    if type != 'all':
        if type == 'functions':
            results = {'functions': results.get('functions', [])}
        elif type == 'classes':
            results = {'classes': results.get('classes', [])}
        elif type == 'modules':
            results = {'modules': results.get('modules', [])}
    
    # Display results
    total_results = sum(len(items) for items in results.values())
    if total_results == 0:
        console.print(Panel(
            f"[yellow]No results found for '{query}'[/yellow]\n\n"
            f"[dim]Try lowering the threshold (current: {threshold})[/dim]\n"
            f"[dim]Or try a different search term[/dim]",
            style="yellow",
            title="🔍 No Results",
            border_style="yellow"
        ))
        return
    
    # Display functions
    if results.get('functions') and (type == 'all' or type == 'functions'):
        table = Table(
            title="🔍 Functions",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            show_lines=True,
            width=120
        )
        
        table.add_column("📦 Package", style="cyan", width=20)
        table.add_column("📁 Module", style="blue", width=25)
        table.add_column("🏷️ Name", style="yellow", width=25)
        table.add_column("📝 Signature", style="green", width=35)
        table.add_column("💯 Score", style="magenta", width=10)
        
        for func in results['functions'][:limit]:
            signature = func.get("signature", "")
            if len(signature) > 35:
                signature = signature[:32] + "..."
                
            table.add_row(
                func.get("package", ""),
                func.get("module", "").split('.')[-1] if func.get("module") else "",
                func.get("name", ""),
                signature,
                f"{func.get('similarity', 0):.3f}"
            )
        
        console.print(table)
    
    # Display classes
    if results.get('classes') and (type == 'all' or type == 'classes'):
        table = Table(
            title="🏛️ Classes",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            show_lines=True,
            width=120
        )
        
        table.add_column("📦 Package", style="cyan", width=20)
        table.add_column("📁 Module", style="blue", width=25)
        table.add_column("🏷️ Name", style="yellow", width=25)
        table.add_column("🔧 Methods", style="green", width=35)
        table.add_column("💯 Score", style="magenta", width=10)
        
        for cls in results['classes'][:limit]:
            methods = cls.get("methods", [])
            methods_str = ", ".join(methods[:3])
            if len(methods) > 3:
                methods_str += f" (+{len(methods)-3})"
            
            table.add_row(
                cls.get("package", ""),
                cls.get("module", "").split('.')[-1] if cls.get("module") else "",
                cls.get("name", ""),
                methods_str,
                f"{cls.get('similarity', 0):.3f}"
            )
        
        console.print(table)
    
    console.print(Panel(
        f"[green]Found {total_results} results with similarity >= {threshold}[/green]\n\n"
        f"[dim]💡 Tips:[/dim]\n"
        f"[dim]• Lower threshold with -t 0.1 for more results[/dim]\n"
        f"[dim]• Use -p package_name to search specific package[/dim]\n"
        f"[dim]• Use -T functions/classes/modules to filter types[/dim]",
        style="green",
        title="🎯 Search Complete",
        border_style="green"
    ))

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 STATS & OTHER COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--package", "-p", help="Show stats for specific package")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed statistics")
@click.pass_context
@handle_errors
def stats(ctx, package: Optional[str], detailed: bool):
    """📊 Show index statistics."""
    
    title = f"📊 Statistics: {package}" if package else "📊 Overall Statistics"
    console.print(create_header(title))
    
    indexer = get_indexer(ctx.obj["db_path"], ctx.obj["verbose"])
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Loading statistics..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Loading...", total=None)
        
        try:
            stats_data = indexer.get_package_stats()
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Failed to load statistics[/red]\n\n"
                f"[dim]Error: {e}[/dim]",
                style="red",
                title="❌ Statistics Error",
                border_style="red"
            ))
            logger.error(f"Failed to load statistics: {e}")
            return
    
    if not stats_data or not stats_data.get('totals'):
        console.print(Panel(
            f"[yellow]📊 Database appears to be empty[/yellow]\n\n"
            f"[dim]Index some packages first:[/dim]\n"
            f"[cyan]adpa index requests[/cyan]",
            style="yellow",
            title="🔍 No Data",
            border_style="yellow"
        ))
        return
    
    # Display overall stats
    if stats_data.get('totals'):
        totals = stats_data['totals']
        
        table = Table(
            title="📊 Overall Statistics",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            width=120
        )
        
        table.add_column("📈 Metric", style="cyan", width=30)
        table.add_column("📊 Value", style="green", width=20)
        
        table.add_row("Total Packages", f"{totals.get('packages', 0):,}")
        table.add_row("Total Functions", f"{totals.get('functions', 0):,}")
        table.add_row("Total Classes", f"{totals.get('classes', 0):,}")
        table.add_row("Total Modules", f"{totals.get('modules', 0):,}")
        
        if detailed:
            table.add_row("Data Classes", f"{totals.get('dataclasses', 0):,}")
            table.add_row("Average Complexity", f"{totals.get('avg_complexity', 0):.2f}")
            table.add_row("Connected Functions", f"{totals.get('connected_functions', 0):,}")
            table.add_row("Total Call Edges", f"{totals.get('total_calls', 0):,}")
        
        console.print(table)

@cli.command(name="call-graph")
@click.option("--source", "-s", help="Source function to analyze")
@click.option("--package", "-p", help="Limit to specific package")
@click.option("--limit", "-n", type=int, default=20, help="Maximum relationships", show_default=True)
@click.pass_context
@handle_errors
def call_graph(ctx, source: Optional[str], package: Optional[str], limit: int):
    """📈 Generate function call graph analysis."""
    
    subtitle_parts = []
    if source:
        subtitle_parts.append(f"Source: {source}")
    if package:
        subtitle_parts.append(f"Package: {package}")
    subtitle_parts.append(f"Limit: {limit}")
    
    console.print(create_header(
        "📈 Call Graph Analysis",
        " | ".join(subtitle_parts)
    ))
    
    indexer = get_indexer(ctx.obj["db_path"], ctx.obj["verbose"])
    
    # Build query
    query_conditions = []
    query_params = []
    
    if source:
        query_conditions.append("source_function LIKE ?")
        query_params.append(f"%{source}%")
    
    if package:
        query_conditions.append("package = ?")
        query_params.append(package)
    
    where_clause = " AND ".join(query_conditions) if query_conditions else "1=1"
    
    call_query = f"""
        SELECT source_function, target_function, package, COUNT(*) as call_count
        FROM call_graph 
        WHERE {where_clause}
        GROUP BY source_function, target_function, package
        ORDER BY call_count DESC, source_function
        LIMIT ?
    """
    query_params.append(limit)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Analyzing call relationships..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Analyzing...", total=None)
        
        try:
            call_results = indexer.conn.execute(call_query, query_params).fetchall()
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Failed to analyze call graph[/red]\n\n"
                f"[dim]Error: {e}[/dim]",
                style="red",
                title="❌ Call Graph Error",
                border_style="red"
            ))
            logger.error(f"Call graph analysis failed: {e}")
            return
    
    if not call_results:
        console.print(Panel(
            f"[yellow]📈 No call relationships found[/yellow]\n\n"
            f"[dim]Try different search criteria or index more packages[/dim]",
            style="yellow",
            title="🔍 No Data",
            border_style="yellow"
        ))
        return
    
    table = Table(
        title="🕸️ Function Call Relationships",
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        show_lines=True,
        width=120
    )
    
    table.add_column("📦 Package", style="cyan", width=20)
    table.add_column("📤 Source Function", style="yellow", width=35)
    table.add_column("📥 Target Function", style="green", width=35)
    table.add_column("📊 Call Count", style="magenta", width=15, justify="right")
    
    for source_func, target_func, pkg, count in call_results:
        table.add_row(pkg, source_func, target_func, str(count))
    
    console.print(table)
    console.print(f"\n[green]Found {len(call_results)} call relationships[/green]")

@cli.command()
@click.option("--force", "-f", is_flag=True, help="Force clean without confirmation")
@click.pass_context
@handle_errors
def clean(ctx, force: bool):
    """🗑️ Clean the index database."""
    
    if not force:
        if not Confirm.ask("⚠️ This will delete all indexed data. Continue?", default=False):
            console.print("[yellow]Cancelled.[/yellow]")
            return
    
    console.print(create_header("🗑️ Cleaning Database"))
    
    indexer = get_indexer(ctx.obj["db_path"], ctx.obj["verbose"])
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Cleaning database..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Cleaning...", total=None)
        
        try:
            indexer.clear_database()
            console.print(Panel(
                "[green]✅ Database cleaned successfully![/green]",
                style="green",
                border_style="green"
            ))
            logger.info("Database cleaned successfully")
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Failed to clean database: {e}[/red]",
                style="red",
                border_style="red"
            ))
            logger.error(f"Failed to clean database: {e}")

if __name__ == "__main__":
    cli()
