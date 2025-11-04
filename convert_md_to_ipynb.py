#!/usr/bin/env python3
"""
Convert markdown files with code blocks to Jupyter notebooks.
This script parses markdown files and creates .ipynb files with proper cell structure.
"""

import json
import re
import sys
from pathlib import Path


def parse_markdown_to_notebook(md_content):
    """
    Parse markdown content and convert to Jupyter notebook format.

    Args:
        md_content: String containing the markdown content

    Returns:
        Dictionary representing a Jupyter notebook
    """
    cells = []

    # Split content by code blocks
    # Pattern: ```python or ``` followed by content and closing ```
    pattern = r'```python\n(.*?)```|```\n(.*?)```|((?:(?!```).)*)'

    # More robust approach: split by ``` markers
    parts = re.split(r'```', md_content)

    i = 0
    while i < len(parts):
        # Odd indices are typically inside code blocks
        if i % 2 == 0:
            # This is markdown text
            text = parts[i].strip()
            if text:
                # Remove 'python' language identifier if it's at the start
                if i + 1 < len(parts) and text.endswith('python'):
                    text = text[:-6].strip()

                if text:
                    cells.append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": text.split('\n')
                    })
        else:
            # This is a code block
            code = parts[i].strip()

            # Check if it starts with a language identifier
            if code.startswith('python\n'):
                code = code[7:]  # Remove 'python\n'
            elif code.startswith('python'):
                code = code[6:].lstrip()

            if code:
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": code.split('\n')
                })

        i += 1

    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook


def convert_md_to_ipynb(md_file_path, output_path=None):
    """
    Convert a markdown file to a Jupyter notebook.

    Args:
        md_file_path: Path to the input markdown file
        output_path: Path for the output .ipynb file (optional)

    Returns:
        Path to the created notebook file
    """
    md_path = Path(md_file_path)

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_file_path}")

    # Read markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert to notebook
    notebook = parse_markdown_to_notebook(md_content)

    # Determine output path
    if output_path is None:
        output_path = md_path.with_suffix('.ipynb')
    else:
        output_path = Path(output_path)

    # Write notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

    return output_path


def main():
    """Main function to handle command line usage."""
    if len(sys.argv) < 2:
        print("Usage: python convert_md_to_ipynb.py <markdown_file> [output_file]")
        print("If output_file is not specified, it will use the same name with .ipynb extension")
        sys.exit(1)

    md_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = convert_md_to_ipynb(md_file, output_file)
        print(f"Successfully converted: {md_file}")
        print(f"Output saved to: {result}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
