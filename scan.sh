#!/bin/bash
# Automated Null-Hound security audit script
# Usage: ./scan.sh <project_name> <source_path> [preset] [--start-from PHASE]
# Example: ./scan.sh webapp /path/to/app nodejs
# Example: ./scan.sh webapp /path/to/app nodejs --start-from audit

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
PROJECT_NAME="$1"
SOURCE_PATH="$2"
PRESET="default"
START_FROM="create"

# Parse remaining arguments
shift 2 2>/dev/null || true
while [ $# -gt 0 ]; do
    case "$1" in
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        *)
            if [ "$PRESET" = "default" ]; then
                PRESET="$1"
            fi
            shift
            ;;
    esac
done

# Check required arguments
if [ -z "$PROJECT_NAME" ] || [ -z "$SOURCE_PATH" ]; then
    log_error "Usage: $0 <project_name> <source_path> [preset] [--start-from PHASE]"
    echo ""
    echo "Arguments:"
    echo "  project_name  - Name for this audit project"
    echo "  source_path   - Path to the codebase to audit"
    echo "  preset        - Optional: php, nodejs, android, or default (default: default)"
    echo "  --start-from  - Optional: create, filter, graph, audit, finalize, report (default: create)"
    echo ""
    echo "Examples:"
    echo "  $0 myapp /home/user/code/app nodejs"
    echo "  $0 myapp /home/user/code/app nodejs --start-from audit"
    echo "  $0 existing-project /path/to/code --start-from finalize"
    echo ""
    echo "Available phases:"
    echo "  create   - Create project (Step 1)"
    echo "  filter   - Generate file filter (Step 2)"
    echo "  graph    - Build knowledge graphs (Step 3)"
    echo "  audit    - Run intuition mode audit (Step 4)"
    echo "  finalize - QA review of findings (Step 5)"
    echo "  report   - Generate HTML report (Step 6)"
    exit 1
fi

# Validate source path exists (skip if starting from finalize/report)
if [ "$START_FROM" != "finalize" ] && [ "$START_FROM" != "report" ]; then
    if [ ! -d "$SOURCE_PATH" ]; then
        log_error "Source path does not exist: $SOURCE_PATH"
        exit 1
    fi
    # Convert to absolute path
    SOURCE_PATH=$(cd "$SOURCE_PATH" && pwd)
fi

log_info "Starting Null-Hound audit"
log_info "Project: $PROJECT_NAME"
log_info "Source: ${SOURCE_PATH:-N/A}"
log_info "Preset: $PRESET"
log_info "Starting from: $START_FROM"
echo ""

# Step 1: Create project
if [ "$START_FROM" = "create" ]; then
    log_info "Step 1/6: Creating project with preset '$PRESET'..."
    if ./hound.py project create "$PROJECT_NAME" "$SOURCE_PATH" "$PRESET"; then
        log_success "Project created successfully"
    else
        log_error "Failed to create project"
        exit 1
    fi
    echo ""
fi

# Step 2: Generate smart file filter
if [ "$START_FROM" = "create" ] || [ "$START_FROM" = "filter" ]; then
    log_info "Step 2/6: Generating intelligent file filter..."
    if ./hound.py filter "$PROJECT_NAME"; then
        log_success "File filter generated"
    else
        log_warning "File filter failed, continuing anyway..."
    fi
    echo ""
fi

# Step 3: Build knowledge graphs
if [ "$START_FROM" = "create" ] || [ "$START_FROM" = "filter" ] || [ "$START_FROM" = "graph" ]; then
    log_info "Step 3/6: Building knowledge graphs..."
    if ./hound.py graph build "$PROJECT_NAME" --iterations 3; then
        log_success "Knowledge graphs built"
    else
        log_error "Failed to build graphs"
        exit 1
    fi
    echo ""
fi

# Step 4: Run intuition mode
if [ "$START_FROM" = "create" ] || [ "$START_FROM" = "filter" ] || [ "$START_FROM" = "graph" ] || [ "$START_FROM" = "audit" ]; then
    log_info "Step 4/6: Running intuition mode audit (deep vulnerability hunting)..."
    if ./hound.py agent audit "$PROJECT_NAME" --mode intuition --time-limit 300; then
        log_success "Intuition mode audit completed"
    else
        log_warning "Intuition mode had issues, continuing to finalization..."
    fi
    echo ""
fi

# Step 5: Finalize findings (QA review)
if [ "$START_FROM" = "create" ] || [ "$START_FROM" = "filter" ] || [ "$START_FROM" = "graph" ] || [ "$START_FROM" = "audit" ] || [ "$START_FROM" = "finalize" ]; then
    log_info "Step 5/6: Finalizing findings (QA review)..."
    if ./hound.py finalize "$PROJECT_NAME"; then
        log_success "Findings finalized"
    else
        log_warning "Finalization had issues, but continuing..."
    fi
    echo ""
fi

# Step 6: Generate report (always run unless we're starting from a later phase)
log_info "Step 6/6: Generating HTML report..."
if ./hound.py report "$PROJECT_NAME"; then
    log_success "Report generated"
else
    log_warning "Report generation had issues"
fi
echo ""

# Display summary
log_success "==================================================="
log_success "Audit completed for: $PROJECT_NAME"
log_success "==================================================="
echo ""
log_info "Next steps:"
echo "  1. Review findings:     ./hound.py project info $PROJECT_NAME"
echo "  2. View report:         open ~/.hound/projects/$PROJECT_NAME/reports/report.html"
echo "  3. List hypotheses:     cat ~/.hound/projects/$PROJECT_NAME/hypotheses_store.json | jq"
echo ""
