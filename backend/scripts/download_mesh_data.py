from pathlib import Path
import urllib.request

MESH_DIR = Path(__file__).resolve().parents[1] / "knowledge" / "mesh"

FILES = {
    "desc2026.xml": "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml",
    "qual2026.xml": "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/qual2026.xml",
    "supp2026.xml": "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/supp2026.xml",
    "pa2026.xml": "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/pa2026.xml",
}


def download_file(filename: str, url: str) -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    target = MESH_DIR / filename

    if target.exists() and target.stat().st_size > 0:
        print(f"Already exists: {target}")
        return

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, target)
    print(f"Saved: {target}")


def main() -> None:
    for filename, url in FILES.items():
        download_file(filename, url)

    print("\nMeSH data download complete.")
    print(f"Files saved under: {MESH_DIR}")


if __name__ == "__main__":
    main()