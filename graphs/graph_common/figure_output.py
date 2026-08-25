import os
from pathlib import Path


OUTPUT_ROOT_ENV = "US_REOPEN_FIGURE_OUTPUT_ROOT"


def figure_artifact_root(figure_name):
    if not figure_name:
        return ""
    figure_name = str(figure_name)
    if not figure_name.startswith("Fig"):
        figure_name = f"Fig{figure_name}"
    return f"graphs/{figure_name}/artifacts"


def configure_output_root(output_root=None, figure=None, default_root=None):
    root = output_root or ""
    if not root and figure:
        root = figure_artifact_root(figure)
    if not root:
        root = os.environ.get(OUTPUT_ROOT_ENV) or ""
    if not root and default_root:
        root = default_root
    if root:
        os.environ[OUTPUT_ROOT_ENV] = root
    return root


def current_output_root():
    return os.environ.get(OUTPUT_ROOT_ENV, "")


def graph_output_path(logical_path, output_root=None):
    logical_path = Path(logical_path)
    root = output_root if output_root is not None else current_output_root()
    if root:
        path = Path(root) / logical_path
    else:
        path = logical_path
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def graph_output_dir(logical_dir, output_root=None):
    path = graph_output_path(Path(logical_dir) / ".keep", output_root=output_root)
    directory = Path(path).parent
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def add_output_args(parser, default_figure=None, default_root=None):
    parser.add_argument(
        "--output_root",
        default=None,
        help=(
            "Root for generated figure artifacts. Logical graph paths are kept "
            "under this root, e.g. graphs/Fig5/artifacts/graphs/graph_start_end/..."
        ),
    )
    parser.add_argument(
        "--figure",
        default=default_figure,
        choices=["Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "1", "2", "3", "4", "5"],
        help="Shortcut for --output_root graphs/FigN/artifacts.",
    )
    parser.set_defaults(_default_output_root=default_root)


def configure_from_args(args):
    return configure_output_root(
        output_root=getattr(args, "output_root", None),
        figure=getattr(args, "figure", None),
        default_root=getattr(args, "_default_output_root", None),
    )
