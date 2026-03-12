from pathlib import Path
import argparse


def build_tree(
    root_dir,
    exclude_dirs=None,
    show_but_do_not_expand=None,
    exclude_files=None,
):
    """
    Build a printable tree of a project directory.

    Parameters
    ----------
    root_dir : str or Path
        Root directory of the project.
    exclude_dirs : set[str] | None
        Directory names to completely ignore.
    show_but_do_not_expand : set[str] | None
        Directory names to show in the tree, but whose contents are not explored.
    exclude_files : set[str] | None
        File names to ignore.

    Returns
    -------
    list[str]
        Tree lines ready to print.
    """
    root = Path(root_dir).resolve()

    exclude_dirs = exclude_dirs or set()
    show_but_do_not_expand = show_but_do_not_expand or set()
    exclude_files = exclude_files or set()

    lines = [root.name]

    def _walk(directory, prefix=""):
        entries = sorted(
            directory.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower())
        )

        visible_entries = []
        for entry in entries:
            if entry.is_dir() and entry.name in exclude_dirs:
                continue
            if entry.is_file() and entry.name in exclude_files:
                continue
            visible_entries.append(entry)

        for index, entry in enumerate(visible_entries):
            is_last = index == len(visible_entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")

            if entry.is_dir() and entry.name not in show_but_do_not_expand:
                extension = "    " if is_last else "│   "
                _walk(entry, prefix + extension)

    _walk(root)
    return lines


def get_relative_paths(
    root_dir,
    exclude_dirs=None,
    show_but_do_not_expand=None,
    exclude_files=None,
):
    """
    Return relative paths from the project root.

    Notes
    -----
    If a directory is in show_but_do_not_expand, the directory itself is returned,
    but its contents are not traversed.
    """
    root = Path(root_dir).resolve()

    exclude_dirs = exclude_dirs or set()
    show_but_do_not_expand = show_but_do_not_expand or set()
    exclude_files = exclude_files or set()

    relative_paths = []

    def _walk(directory):
        entries = sorted(
            directory.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower())
        )

        for entry in entries:
            if entry.is_dir() and entry.name in exclude_dirs:
                continue
            if entry.is_file() and entry.name in exclude_files:
                continue

            relative_paths.append(entry.relative_to(root))

            if entry.is_dir() and entry.name not in show_but_do_not_expand:
                _walk(entry)

    _walk(root)
    return relative_paths


def main():
    parser = argparse.ArgumentParser(
        description="Print the tree of a project directory."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory of the project (default: current directory)."
    )
    parser.add_argument(
        "--exclude-dirs",
        nargs="*",
        default=["__pycache__", ".git", ".idea",],
        help="Directory names to hide completely."
    )
    parser.add_argument(
        "--show-no-expand",
        nargs="*",
        default=["venv", ".venv_build", "dist", "build"],
        help="Directory names to show but not traverse."
    )
    parser.add_argument(
        "--exclude-files",
        nargs="*",
        default=[],
        help="File names to ignore."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output text file."
    )

    args = parser.parse_args()

    tree_lines = build_tree(
        root_dir=args.root,
        exclude_dirs=set(args.exclude_dirs),
        show_but_do_not_expand=set(args.show_no_expand),
        exclude_files=set(args.exclude_files),
    )

    result = "\n".join(tree_lines)
    print(result)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()