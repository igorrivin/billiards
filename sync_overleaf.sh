#!/bin/bash
# Script to sync between local paper directory and Overleaf

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Overleaf Sync Tool${NC}"
echo "==================="

case "$1" in
  "pull")
    echo -e "${GREEN}Pulling from Overleaf...${NC}"
    # Fetch latest from Overleaf
    git fetch overleaf
    
    # Save current state
    cp paper/billiard_orbits.tex paper/billiard_orbits_local_backup.tex
    
    # Get files from Overleaf
    git show overleaf/master:billiard_orbits.tex > paper/billiard_orbits.tex
    git show overleaf/master:references.bib > paper/references.bib
    git show overleaf/master:grid_orbits.png > paper/grid_orbits.png 2>/dev/null || echo "No grid_orbits.png in Overleaf"
    
    echo "Files pulled from Overleaf to paper/"
    echo "Local backup saved as paper/billiard_orbits_local_backup.tex"
    ;;
    
  "push")
    echo -e "${GREEN}Pushing to Overleaf...${NC}"
    # This is more complex - we need to use subtree push
    git add paper/
    git commit -m "Update paper files for Overleaf sync"
    git subtree push --prefix=paper overleaf master
    echo "Files pushed to Overleaf"
    ;;
    
  "diff")
    echo -e "${GREEN}Showing differences...${NC}"
    git fetch overleaf
    echo "=== Differences in billiard_orbits.tex ==="
    git show overleaf/master:billiard_orbits.tex | diff -u - paper/billiard_orbits.tex || true
    ;;
    
  *)
    echo "Usage: $0 {pull|push|diff}"
    echo "  pull - Get latest changes from Overleaf"
    echo "  push - Push your local changes to Overleaf"
    echo "  diff - Show differences between local and Overleaf"
    exit 1
    ;;
esac