import os
from pathlib import Path

# -----------------------------------------------------------------------------
# This script programmatically creates the file and folder structure
# for the Credit Card Fraud Detection project, including docs folder.
# -----------------------------------------------------------------------------

def create_project_structure(root_dir):
    """
    Creates the project directory and file structure.

    Args:
        root_dir (str): The name of the root project directory.
    """
    project_structure = {
        'data': {
            'raw': ['.gitkeep'],           # Keeps folder in git, no placeholder dataset
            'processed': ['.gitkeep']
        },
        'notebooks': [
            '.gitkeep'
        ],
        'src': [
            '__init__.py',
            'preprocess_data.py',
            'train_model.py',
            'predict.py'
        ],
        'models': ['.gitkeep'],            # Keeps folder in git, no placeholder model
        'docs': [                          # Documentation folder
            'FILE_STRUCTURE.md',
            'WORKFLOW.md'
        ],
        'README.md': None,
        'requirements.txt': None,
        '.gitignore': None,
    }

    project_path = Path(root_dir)
    project_path.mkdir(exist_ok=True)
    print(f"Created project root directory: {project_path}")

    def create_items(base_path, structure):
        for name, content in structure.items():
            current_path = base_path / name

            if content is None:  # It's a file
                current_path.touch()
                print(f"  Created file: {current_path}")

            elif isinstance(content, list):  # It's a directory with files
                current_path.mkdir(parents=True, exist_ok=True)
                print(f"  Created directory: {current_path}")
                for file_name in content:
                    file_path = current_path / file_name
                    file_path.touch()
                    print(f"    Created file: {file_path}")

            elif isinstance(content, dict):  # It's a directory with subdirectories
                current_path.mkdir(parents=True, exist_ok=True)
                print(f"  Created directory: {current_path}")
                create_items(current_path, content)

            else:
                print(f"Skipped unknown item type: {name}")


    create_items(project_path, project_structure)
    print("\n✅ Project structure created successfully!")


if __name__ == "__main__":
    project_name = "003-Credit_Card_Fraud_Detection"
    create_project_structure(project_name)
