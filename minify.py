import re
import sys
import os

def minify_python(code):
    """Minifie un code Python en supprimant les commentaires et espaces inutiles."""
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
    """Minifie un code CSS en supprimant les espaces et commentaires."""
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Supprime les commentaires
    code = re.sub(r'\s+', ' ', code)  # Remplace plusieurs espaces par un seul
    code = re.sub(r'\s*([{};,:])\s*', r'\1', code)  # Supprime les espaces autour de {} ; : ,
    return code.strip()

def minify_js(code):
    """Minifie un code JavaScript en supprimant les espaces et commentaires."""
    code = re.sub(r'//.*', '', code)  # Supprime les commentaires ligne
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Supprime les commentaires multi-lignes
    code = re.sub(r'\s+', ' ', code)  # Remplace plusieurs espaces par un seul
    code = re.sub(r'\s*([{};,:])\s*', r'\1', code)  # Supprime les espaces autour de {} ; : ,
    return code.strip()

def minify_html(code):
    """Minifie un code HTML en supprimant les espaces et commentaires."""
    code = re.sub(r'<!--.*?-->', '', code, flags=re.DOTALL)  # Supprime les commentaires
    code = re.sub(r'\s+', ' ', code)  # Remplace plusieurs espaces par un seul
    return code.strip()

def detect_and_minify(input_file, output_file):
    """Détecte le type de fichier et applique la minification correspondante."""
    if not os.path.exists(input_file):
        print(f"Erreur : Le fichier {input_file} n'existe pas.")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        code = f.read()

    file_ext = os.path.splitext(input_file)[1].lower()

    if file_ext == ".py":
        print("🔹 Détection : Python")
        minified_code = minify_python(code)
    elif file_ext == ".css":
        print("🔹 Détection : CSS")
        minified_code = minify_css(code)
    elif file_ext == ".js":
        print("🔹 Détection : JavaScript")
        minified_code = minify_js(code)
    elif file_ext == ".html":
        print("🔹 Détection : HTML")
        minified_code = minify_html(code)
    else:
        print("❌ Format non pris en charge.")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(minified_code)

    print(f"✅ Optimisation terminée ! Fichier sauvegardé sous : {output_file}")

# Vérification des arguments
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Utilisation : python minify.py input_file output_file")
    else:
        detect_and_minify(sys.argv[1], sys.argv[2])
