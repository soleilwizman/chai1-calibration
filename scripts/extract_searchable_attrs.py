import json
from pathlib import Path


def walk_schema(node, prefix="", searchable=None):
    if searchable is None:
        searchable = []
    if not isinstance(node, dict):
        return searchable
    for key, value in (node.get("properties") or {}).items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value.get("type") == "array" and "items" in value:
            walk_schema(value["items"], path, searchable)
        elif isinstance(value, dict) and value.get("properties"):
            walk_schema(value, path, searchable)
        elif isinstance(value, dict) and value.get("rcsb_search_context"):
            searchable.append(path)
    return searchable


def main():
    schema_path = Path(__file__).resolve().parents[1] / "schema.json"
    output_path = Path(__file__).resolve().parents[1] / "scripts" / "searchable_attrs.txt"

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    attrs = sorted(set(walk_schema(schema)))

    output_path.write_text("\n".join(attrs) + "\n", encoding="utf-8")
    print(f"Wrote {len(attrs)} searchable attributes to {output_path}")


if __name__ == "__main__":
    main()
