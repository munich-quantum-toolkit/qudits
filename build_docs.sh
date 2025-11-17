#!/usr/bin/env bash

# Build and serve MQT Qudits documentation locally
# This script builds the Sphinx documentation and serves it on a local HTTP server

set -e

echo "🔨 Building MQT Qudits Documentation..."
echo "========================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating one...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

# Install documentation dependencies
echo -e "${BLUE}📦 Installing documentation dependencies...${NC}"
pip install -q -e ".[docs]"
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Clean previous builds
echo -e "${BLUE}🧹 Cleaning previous builds...${NC}"
rm -rf docs/_build
echo -e "${GREEN}✓ Cleaned${NC}"

# Build the documentation
echo -e "${BLUE}📚 Building HTML documentation...${NC}"
cd docs

# Run sphinx-build
python -m sphinx -b html . _build/html -W --keep-going

cd ..

echo ""
echo -e "${GREEN}✅ Documentation built successfully!${NC}"
echo ""
echo -e "${BLUE}📖 Documentation location: ${NC}docs/_build/html/index.html"
echo ""
echo -e "${YELLOW}🚀 Starting local server...${NC}"
echo -e "${BLUE}📡 Server will be available at: ${GREEN}http://localhost:8000${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start local HTTP server
cd docs/_build/html
python -m http.server 8000
