from pathlib import Path
import json
from typing import Dict, Union

def get_directory_structure(startpath: Path) -> Dict[str, Union[str, dict]]:
    """
    Convert directory structure to a dictionary format
    
    Args:
        startpath (Path): The root directory to start from
        
    Returns:
        Dict: Directory structure as a nested dictionary
    """
    if not startpath.exists():
        return {"error": "Directory does not exist"}
        
    structure = {"name": startpath.name, "type": "directory", "children": []}
    
    for item in sorted(startpath.iterdir()):
        if item.is_dir():
            structure["children"].append(get_directory_structure(item))
        else:
            structure["children"].append({
                "name": item.name,
                "type": "file"
            })
    
    return structure

def save_directory_structure(structure: dict, output_file: str = "data_structure.json") -> None:
    """
    Save the directory structure to a JSON file
    
    Args:
        structure (dict): Directory structure dictionary
        output_file (str): Output JSON file path
    """
    output_path = Path(__file__).parent.parent.parent / "data" / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(structure, f, indent=2, ensure_ascii=False)
    print(f"Directory structure saved to: {output_path}")

def main():
    data_dir = Path(__file__).parent.parent.parent / "data"
    structure = get_directory_structure(data_dir)
    save_directory_structure(structure)

if __name__ == "__main__":
    main()