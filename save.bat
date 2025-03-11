@echo off
cd /d "C:\chemin\vers\ton\projet"
git add .
git commit -m "Auto-update: %date% %time%"
git push origin main
echo ✅ Projet sauvegardé !
