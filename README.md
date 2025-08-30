<div align="center">
  <img src="static/hound.png" alt="Hound" width="450" />
  
  # Hound
  
  **Autonomous agents for code security auditing**
  
  [![Tests](https://github.com/muellerberndt/hound/workflows/Tests/badge.svg)](https://github.com/muellerberndt/hound/actions)
  [![License: Apache 2.0](LICENSE.md)](https://www.apache.org/licenses/LICENSE-2.0)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
  [![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-74aa9c)](https://openai.com)
  [![Gemini](https://img.shields.io/badge/Gemini-Compatible-4285F4)](https://ai.google.dev/)
  [![Anthropic](https://img.shields.io/badge/Anthropic-Compatible-6B46C1)](https://anthropic.com)
  
</div>

---

## Overview

Hound is a security audit automation pipeline for AI-assisted code review that mirrors how expert auditors think, learn, and collaborate. Read the [blog post](https://muellerberndt.medium.com/unleashing-the-hound-how-ai-agents-find-deep-logic-bugs-in-any-codebase-64c2110e3a6f) to learn more.

### How It Works

Hound's cognitive architecture mirrors how expert auditors think:

1. **Relational Knowledge Graphs** - Builds interconnected graphs that capture system aspects (architecture, access control, value flows) with annotations marking observations and assumptions. This allows Hound to improve its understanding of the codebase over time and detect contradictions between high-level assumptions and implementation details.

2. **Precise Code Grounding** - Every graph node and annotation links to exact code locations. Instead of vague semantic matching with embeddings, Hound maintains attention on the exact specific functions, variables, and call sites that are relevant to its current investigation.

3. **Adaptive Planning** - Investigation priorities dynamically reorganize based on discoveries. Finding one vulnerability triggers focused searches for related issues. Coverage tracking ensures systematic exploration while allowing strategic pivots.

The system employs a **senior/junior agent pattern**: a Strategist (senior) reviews graphs to identify contradictions and plan investigations, while a Scout (junior) executes targeted code analysis. This mirrors real audit teams where seniors direct while juniors investigate.

Each audit creates a **session** that tracks coverage, findings, and token usage. Knowledge accumulates across sessions, building deeper understanding over time.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set up your API keys:

```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here  
export GOOGLE_API_KEY=your_key_here
```

Configure models in `config.yaml`:

```yaml
models:
  scout:      # Junior agent for exploration
    platform: openai
    model: gpt-4o-mini
  
  strategist: # Senior agent for planning
    platform: openai
    model: gpt-4o
    
  finalizer:  # Report generation
    platform: openai
    model: gpt-4o
```

## Audit Workflow Walkthrough

### Step 1: Create a Project

Projects organize your audits and store all analysis data:

```bash
# Create a project from local code
./hound.py project create myaudit /path/to/code

# List all projects
./hound.py project list

# View project details and coverage
./hound.py project info myaudit
```

### Step 2: Build Knowledge Graphs

Hound analyzes your codebase and builds aspect-oriented knowledge graphs:

```bash
# Build graphs (uses scout model by default)
./hound.py graph build myaudit

# Customize graph types and depth
./hound.py graph build myaudit --graphs 5 --iterations 3

# View generated graphs
./hound.py graph list myaudit
```

**What happens:** Hound inspects the codebase and creates graphs for different aspects:
- **SystemArchitecture** - Components and their interactions
- **AuthorizationAndAccessControl** - Permission systems and role management
- **MonetaryFlows** - Value transfers and financial operations
- **DataFlowAndState** - State variables and data transformations
- **ExternalInterfaces** - External calls and integrations

Each graph links abstract concepts to specific code locations, building a semantic understanding of your system.

### Step 3: Run the Audit

The audit phase uses the **senior/junior pattern** with planning and investigation:

```bash
# Run a full audit with strategic planning
./hound.py agent audit myaudit

# Customize planning and investigation depth
./hound.py agent audit myaudit --plan-n 5 --iterations 10

# Use specific models
./hound.py agent audit myaudit \
  --model gpt-4o-mini \
  --strategist-model gpt-4o

# Enable debug logging (captures all prompts/responses)
./hound.py agent audit myaudit --debug
```

**What happens during an audit:**

The audit is a **dynamic, iterative process** with continuous interaction between Strategist and Scout:

1. **Initial Planning** (Strategist)
   - Reviews all knowledge graphs and annotations
   - Identifies contradictions between assumptions and observations
   - Creates a batch of prioritized investigations (default: 5)
   - Focus areas: access control violations, value transfer risks, state inconsistencies

2. **Investigation Loop** (Scout + Strategist collaboration)
   
   For each investigation in the batch:
   - **Scout explores**: Loads relevant graph nodes, analyzes code
   - **Scout escalates**: When deep analysis needed, calls Strategist via `deep_think`
   - **Strategist analyzes**: Reviews Scout's collected context, forms vulnerability hypotheses
   - **Hypotheses form**: Findings are added to global store
   - **Coverage updates**: Tracks visited nodes and analyzed code

3. **Adaptive Replanning**
   
   After completing a batch:
   - Strategist reviews new findings and updated coverage
   - Reorganizes priorities based on discoveries
   - If vulnerability found, searches for related issues
   - Plans next batch of investigations
   - Continues until coverage goals met or no promising leads remain

4. **Session Management**
   - Unique session ID tracks the entire audit lifecycle
   - Coverage metrics show exploration progress
   - All findings accumulate in hypothesis store
   - Token usage tracked per model and investigation

**Example output:**
```
Planning Next Investigations...
1. [P10] Investigate role management bypass vulnerabilities
2. [P9] Check for reentrancy in value transfer functions
3. [P8] Analyze emergency function privilege escalation

Coverage Statistics:
  Nodes visited: 23/45 (51.1%)
  Cards analyzed: 12/30 (40.0%)

Hypotheses Status:
  Total: 15
  High confidence: 8
  Confirmed: 3
```

### Step 4: Run Targeted Investigations

For specific concerns, run focused investigations without full planning:

```bash
# Investigate a specific concern
./hound.py agent investigate "Check for reentrancy in withdraw function" myaudit

# Quick investigation with fewer iterations
./hound.py agent investigate "Analyze access control in admin functions" myaudit \
  --iterations 5
```

### Step 5: Finalize and Generate Reports

Review findings and produce professional audit reports:

```bash
# Generate comprehensive audit report
./hound.py finalize myaudit

# Customize report generation
./hound.py finalize myaudit \
  --confidence-threshold 0.7 \
  --model gpt-4o

# View the generated report
./hound.py report view myaudit
```

**Report includes:**
- Executive summary
- System architecture overview
- Detailed findings with severity ratings
- Code references and attack scenarios
- Remediation recommendations

## Session Management

Each audit run creates a session with comprehensive tracking:

```bash
# View session details
./hound.py project info myaudit

# Session data includes:
# - Coverage statistics (nodes/cards visited)
# - Investigation history
# - Token usage by model
# - Planning decisions
# - Hypothesis formation
```

Sessions are stored in `~/.hound/projects/myaudit/sessions/` and contain:
- `session_id`: Unique identifier
- `coverage`: Visited nodes and analyzed code
- `investigations`: All executed investigations
- `planning_history`: Strategic decisions made
- `token_usage`: Detailed API usage metrics

## Managing Hypotheses

Hypotheses accumulate across sessions as the agent learns:

```bash
# List all hypotheses with confidence scores
./hound.py project hypotheses myaudit

# View detailed hypothesis information
./hound.py project hypotheses myaudit --details

# Reset hypotheses (creates backup)
./hound.py project reset-hypotheses myaudit

# Force reset without confirmation
./hound.py project reset-hypotheses myaudit --force
```

## Advanced Features

### Model Selection

Override default models per component:

```bash
# Use different models for each role
./hound.py agent audit myaudit \
  --platform openai --model gpt-4o-mini \           # Scout
  --strategist-platform anthropic --strategist-model claude-3-opus \  # Strategist
  --finalizer-platform openai --finalizer-model gpt-4o  # Finalizer
```

### Debug Mode

Capture all LLM interactions for analysis:

```bash
# Enable debug logging
./hound.py agent audit myaudit --debug

# Debug logs saved to .hound_debug/
# Includes HTML reports with all prompts and responses
```

### Coverage Tracking

Monitor audit progress and completeness:

```bash
# View coverage statistics
./hound.py project coverage myaudit

# Coverage shows:
# - Graph nodes visited vs total
# - Code cards analyzed vs total
# - Percentage completion
```

### Parallel Execution

Run multiple investigations concurrently:

```bash
# Run parallel investigations (experimental)
./hound.py agent audit myaudit --parallel --workers 3
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.