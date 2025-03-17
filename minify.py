import os
import re

def minify_python(code):
    """Minifie un fichier Python en supprimant les commentaires et espaces inutiles."""
    lines = code.split("\n")
    minified_lines = []
    for line in lines:
        line = line.strip()  # Supprime les espaces autour
        if not line or line.startswith("#"):  # Ignore les lignes vides et commentaires
            continue
        line = re.sub(r"\s*#.*", "", line)  # Supprime les commentaires en fin de ligne
        line = re.sub(r"\s+", " ", line)  # Remplace plusieurs espaces par un seul
        minified_lines.append(line)
    return ";".join(minified_lines)  # Évite les sauts de ligne inutiles

def minify_css(code):
    """Minifie un fichier CSS en supprimant les espaces et commentaires."""
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Supprime les commentaires
    code = re.sub(r'\s+', ' ', code)  # Remplace plusieurs espaces par un seul
    code = re.sub(r'\s*([{};,:])\s*', r'\1', code)  # Supprime les espaces autour de {} ; : ,
    return code.strip()

def process_file(file_path, minify_func, new_extension):
    """Minifie un fichier et remplace l'ancienne version minifiée."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        minified_code = minify_func(code)
        min_file_path = f"{file_path[:-len(new_extension)]}.min{new_extension}"

        # Vérifier si le fichier minifié existe déjà et s'il a changé
        if os.path.exists(min_file_path):
            with open(min_file_path, 'r', encoding='utf-8') as f:
                existing_code = f.read()
            if existing_code == minified_code:
                print(f"⏩ Pas de changement : {min_file_path}")
                return  # Ne réécrit pas si identique

        with open(min_file_path, 'w', encoding='utf-8') as f:
            f.write(minified_code)
        print(f"✅ Minifié : {min_file_path}")

    except Exception as e:
        print(f"❌ Erreur sur {file_path}: {e}")

def clean_old_minified(root_dir):
    """Supprime les anciens fichiers minifiés avant de recréer les nouveaux."""
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".min.py") or filename.endswith(".min.css"):
                file_path = os.path.join(foldername, filename)
                os.remove(file_path)
                print(f"🗑️ Supprimé : {file_path}")

def minify_project(root_dir):
    """Parcourt récursivement un dossier et minifie tous les fichiers .py et .css."""
    clean_old_minified(root_dir)  # Supprime les anciens fichiers minifiés

    for foldername, subfolders, filenames in os.walk(root_dir):
        # Exclure les dossiers __pycache__
        if "__pycache__" in foldername:
            continue

        for filename in filenames:
            file_path = os.path.join(foldername, filename)

            if filename.endswith(".py") and not filename.endswith(".min.py"):
                process_file(file_path, minify_python, ".py")
            elif filename.endswith(".css") and not filename.endswith(".min.css"):
                process_file(file_path, minify_css, ".css")

# Exécution
if __name__ == "__main__":
    root_directory = os.getcwd()  # Dossier du projet
    print(f"🚀 Minification du projet dans : {root_directory}")
    minify_project(root_directory)
    print("✅ Minification terminée !")
