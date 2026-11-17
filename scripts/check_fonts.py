"""Checking Windows fonts for rendering."""
import os

FONTS = r"C:\Windows\Fonts"
for f in ["consola.ttf", "consolab.ttf", "segoeui.ttf", "segoeuib.ttf", "arialbd.ttf", "arial.ttf"]:
    print(f, os.path.exists(os.path.join(FONTS, f)))
