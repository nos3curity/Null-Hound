<div align="center">
  <img src="static/hound.png" alt="Hound" width="450" />
  
  # Hound
  
  **Autonomous agents for code security auditing**
  
  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
  [![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-74aa9c)](https://openai.com)
  [![Gemini](https://img.shields.io/badge/Gemini-Compatible-4285F4)](https://ai.google.dev/)
  [![Anthropic](https://img.shields.io/badge/Anthropic-Compatible-6B46C1)](https://anthropic.com)
  
</div>

---

## Overview

## Overview

**Hound** is a research framework for **AI-driven code security auditing** that mirrors how expert auditors actually think, learn, and collaborate. Instead of spamming shallow hypotheses or rigidly parsing syntax trees, Hound builds **flexible cognitive models** of a system that accumulate, adapt, and get refined over time.  

Agents reason across **abstract business logic and concrete code details**, capturing assumptions, invariants, and observations into evolving **knowledge graphs**. These graphs are dynamic relational models that change as the audit deepens. The process mirrors real audit teams: weeks of cumulative understanding, targeted hypotheses, and collaborative review that surface **deep, business-logic vulnerabilities** missed by pattern-matching tools.

### Key innovations

- **Dynamic modeling** of any codebase, from small libraries to complex protocols  
- **Aspect graphs** that relate abstract concepts (monetary flows, authorization, invariants) to concrete implementations (functions, storage, calls)  
- **Iterative accumulation** of knowledge — beliefs, hypotheses, and observations evolve with time, not discarded after each run  
- **Dynamic model switching**: lightweight agents can escalate reasoning to larger models for guidance and hypothesis formation  
- **Collaborative orchestration**: run multiple agents in parallel or serial pipelines, mirroring real audit teams  
- **Professional outputs**: generate complete audit reports with executive summaries, system overviews, and detailed findings  

Hound is designed to scale with both **time and resources**: a one-hour run gives quick coverage, while a days-long review provide more detailed results.

 See the [tech overview](tech.md) for details.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start - Security Audit Workflow

### 1. Set up your API key

```bash
export OPENAI_API_KEY="your-api-key"
# or
export GOOGLE_API_KEY="your-api-key"  # for Gemini
```

### 2. Configure Hound

```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your preferred models
```

### 3. Create a project

```bash
# Create a new project from source code
python hound.py project create my_audit /path/to/source/code

# Filter by main projet files (useful to skip tests and interfaces)
python hound.py project create my_audit /path/to/source/code --files "file1,file2"

# View project details
python hound.py project info my_audit
```

### 4. Build knowledge graphs

```bash
# Generate system architecture graphs (analyzes code structure)
python hound.py graph build my_audit --graphs 3 --iterations 5

# Export for visualization
python hound.py graph export my_audit --output graphs.html
```

### 5. Run security audit

```bash
# Explore and investigate the codebase
python hound.py agent audit my_audit --time-limit 3

# The audit agent will:
# - Analyze the knowledge graphs
# - Investigate potential security issues
# - Form hypotheses about vulnerabilities
# - Update graphs with verified observations and assumptions
```

### 6. Finalize high-confidence findings

```bash
# Review and confirm high-confidence hypotheses
python hound.py agent finalize my_audit

# This step:
# - Reviews hypotheses with confidence >= 0.7
# - Performs deeper validation
# - Confirms or rejects findings
# - Updates confidence levels
```

### 7. Generate audit report

```bash
# Create security audit report
python hound.py report my_audit --output report.html

# The report includes:
# - Executive summary
# - Confirmed vulnerabilities
# - Risk assessments
# - Detailed findings with code locations
```

### 8. Additional analysis options

```bash
# Add custom graphs with specific focus
python hound.py graph add-custom --project my_project \
    --focus "Access control implementation"

# Investigate specific questions
python hound.py agent investigate "Check for SQL injection vulnerabilities" \
    --project my_audit --iterations 10

# View project hypotheses with confidence ratings
python hound.py project hypotheses my_audit
```

## Commands

### Project Management
- `project create` - Create a new project
- `project list` - List all projects
- `project info` - Show project information including hypotheses
- `project hypotheses` - List all hypotheses with confidence ratings
- `project delete` - Delete a project

### Graph Operations
- `graph build` - Build system architecture graphs
- `graph add-custom` - Add custom graph with user-defined focus
- `graph export` - Export graphs to interactive HTML visualization

### Agent Commands
- `agent audit` - Run comprehensive security audit and form hypotheses
- `agent finalize` - Validate and confirm high-confidence findings
- `agent investigate` - Run targeted investigation with specific prompt

### Reporting
- `report` - Generate professional HTML security audit report

## Getting Help

```bash
# General help
python hound.py --help

# Command group help
python hound.py project --help
python hound.py graph --help
python hound.py agent --help

# Command-specific help
python hound.py graph build --help
```

## Configuration

See `config.yaml.example` for all available options and model configurations.

## Contributing

Contributions are welcome! To contribute to the base framework, you need to sign the [Contributor License Agreement](https://cla-assistant.io/muellerberndt/hound).
