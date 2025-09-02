<div align="center">
  <img src="static/hound.png" alt="Hound" width="450" />
  
  # Hound
  
  **Autonomous agents for code security auditing**
  
  [![Tests](https://github.com/muellerberndt/hound/workflows/Tests/badge.svg)](https://github.com/muellerberndt/hound/actions)
  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
  [![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-74aa9c)](https://openai.com)
  [![Gemini](https://img.shields.io/badge/Gemini-Compatible-4285F4)](https://ai.google.dev/)
  [![Anthropic](https://img.shields.io/badge/Anthropic-Compatible-6B46C1)](https://anthropic.com)
  
</div>

---

## Overview

Hound is a security audit automation pipeline for AI-assisted code review that mirrors how expert auditors think, learn, and collaborate. 

### How It Works

Hound's cognitive architecture mirrors how expert auditors think:

1. **Relational Knowledge Graphs** - Builds interconnected graphs that capture system aspects (architecture, access control, value flows) with annotations marking observations and assumptions. This allows Hound to improve its understanding of the codebase over time and detect contradictions between high-level assumptions and implementation details.

2. **Precise Code Grounding** - Every graph node and annotation links to exact code locations. Instead of vague semantic matching with embeddings, Hound maintains attention on the exact specific functions, variables, and call sites that are relevant to its current investigation.

3. **Adaptive Planning** - Investigation priorities dynamically reorganize based on discoveries. Finding one vulnerability triggers focused searches for related issues. Coverage tracking ensures systematic exploration while allowing strategic pivots.

The system employs a **senior/junior agent pattern**: a Strategist (senior) reviews graphs to identify contradictions and plan investigations, while a Scout (junior) executes targeted code analysis. This mirrors real audit teams where seniors direct while juniors investigate.

Each audit creates a **session** that tracks coverage, findings, and token usage. Knowledge accumulates across sessions, building deeper understanding over time.

### Links

- [Original blog post](https://muellerberndt.medium.com/unleashing-the-hound-how-ai-agents-find-deep-logic-bugs-in-any-codebase-64c2110e3a6f)
- [Hound benchmarking pipeline](https://github.com/muellerberndt/hound-empirical-development)
- [Smart contract audit benchmarks datasets and tooling](https://github.com/muellerberndt/scabench)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set up your API keys, e.g.:

```bash
export OPENAI_API_KEY=your_key_here
```

Configure models in `config.yaml`:

```yaml

graph:
    platform: openai
    model: gpt-4.1

models:
  scout:      # Junior auditor
    platform: openai
    model: gpt-5-mini
  
  strategist: # Senior auditopr
    platform: openai
    model: gpt-5
    reasoning_effort: high
    text_verbosity: low
    
  finalizer:  # Report generation
    platform: openai
    model: gpt-5
```

## Complete Audit Workflow

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

Hound inspects the codebase and creates graphs for different aspects. Each graph links abstract concepts to specific code locations, building a semantic understanding of your system.

### Step 3: Run the Audit

The audit phase uses the **senior/junior pattern** with planning and investigation:

```bash
# Run a full audit with strategic planning (new session)
./hound.py agent audit myaudit

# Use specific models
./hound.py agent audit myaudit \
  --model gpt-4o-mini \
  --strategist-model gpt-4o

# Enable debug logging (captures all prompts/responses)
./hound.py agent audit myaudit --debug

# Attach to an existing session and continue where you left off
./hound.py agent audit myaudit --session <session_id>
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

### Step 4: Monitor Progress

Check audit progress and findings at any time:

```bash
# View current hypotheses (findings)
./hound.py hypotheses list myaudit

# See detailed hypothesis information
./hound.py hypotheses list myaudit --verbose

# Filter by confidence level
./hound.py hypotheses list myaudit --min-confidence 0.8

# Check coverage statistics
./hound.py project coverage myaudit

# View session details
./hound.py project info myaudit
```

### Step 5: Run Targeted Investigations

For specific concerns, run focused investigations without full planning:

```bash
# Investigate a specific concern
./hound.py agent investigate "Check for reentrancy in withdraw function" myaudit

# Quick investigation with fewer iterations
./hound.py agent investigate "Analyze access control in admin functions" myaudit \
  --iterations 5
```

### Step 6: Finalize Findings

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

### Step 7: Generate Proof-of-Concepts

Create and manage proof-of-concept exploits for confirmed vulnerabilities:

```bash
# Generate PoC prompts for confirmed vulnerabilities
./hound.py poc make-prompt myaudit

# Generate for a specific hypothesis
./hound.py poc make-prompt myaudit --hypothesis hyp_12345

# Import existing PoC files
./hound.py poc import myaudit hyp_12345 exploit.sol test.js \
  --description "Demonstrates reentrancy exploit"

# List all imported PoCs
./hound.py poc list myaudit
```

The PoC workflow:
1. **make-prompt** generates detailed prompts for coding agents (like Claude Code) to create exploits
2. **import** adds your PoC files to the project, linking them to specific vulnerabilities
3. Imported PoCs are automatically included in the final report with syntax highlighting

### Step 8: Generate Professional Reports

Produce comprehensive audit reports with all findings and PoCs:

```bash
# Generate HTML report (includes imported PoCs)
./hound.py report myaudit

# View the generated report
./hound.py report view myaudit

# Export report to specific location
./hound.py report myaudit --output /path/to/report.html
```

Reports include:
- Executive summary
- Detailed vulnerability descriptions
- Affected code snippets
- Proof-of-concept exploits (if imported)
- Severity ratings and confidence scores

## Complete Example Workflow

Here's a full audit from start to finish:

```bash
# 1. Create project
./hound.py project create myaudit /path/to/code

# 2. Build knowledge graphs
./hound.py graph build myaudit --graphs 5 --iterations 3

# 3. Run initial audit
./hound.py agent audit myaudit --time-limit 30

# 4. Check findings
./hound.py hypotheses list myaudit

# 5. Continue audit if needed
./hound.py agent audit myaudit --session <session_id> --time-limit 20

# 6. Finalize high-confidence findings
./hound.py finalize myaudit --confidence-threshold 0.7

# 7. Generate PoC prompts
./hound.py poc make-prompt myaudit

# 8. Import PoCs (after creating them)
./hound.py poc import myaudit hyp_abc123 exploit.sol

# 9. Generate final report
./hound.py report myaudit

# 10. View results
./hound.py report view myaudit
```

## Session Management

Each audit run operates under a session with comprehensive tracking and per-session planning:

- Planning is stored in a per-session PlanStore with statuses: `planned`, `in_progress`, `done`, `dropped`, `superseded`.
- Existing `planned` items are executed first; Strategist only tops up new items to reach your `--plan-n`.
- On resume, any stale `in_progress` items are reset to `planned`; completed items remain `done` and are not duplicated.
- Completed investigations, coverage, and hypotheses are fed back into planning to avoid repeats and guide prioritization.

```bash
# View session details
./hound.py project info myaudit

# List and inspect sessions
./hound.py project sessions myaudit --list
./hound.py project sessions myaudit <session_id>

# Show planned investigations for a session (Strategist PlanStore)
./hound.py project plan myaudit <session_id>

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

Resume/attach to an existing session during an audit run by passing the session ID:

```bash
# Attach to a specific session and continue auditing under it
./hound.py agent audit myaudit --session <session_id>
```

When you attach to a session, its status is set to `active` while the audit runs and finalized on completion (`completed` or `interrupted` if a time limit was hit). Any `in_progress` plan items are reset to `planned` so you can continue cleanly.

### Simple Planning Examples

```bash
# Start an audit (creates a session automatically)
./hound.py agent audit myaudit

# List sessions to get the session id
./hound.py project sessions myaudit --list

# Show planned investigations for that session
./hound.py project plan myaudit <session_id>

# Attach later and continue planning/execution under the same session
./hound.py agent audit myaudit --session <session_id>
```

## Managing Hypotheses

Hypotheses are the core findings that accumulate across sessions:

```bash
# List all hypotheses with confidence scores
./hound.py hypotheses list myaudit

# View with full details
./hound.py hypotheses list myaudit --verbose

# Filter by status or confidence
./hound.py hypotheses list myaudit --status confirmed
./hound.py hypotheses list myaudit --min-confidence 0.8

# Update hypothesis status
./hound.py hypotheses update myaudit hyp_12345 --status confirmed

# Reset hypotheses (creates backup)
./hound.py hypotheses reset myaudit

# Force reset without confirmation
./hound.py hypotheses reset myaudit --force
```

Hypothesis statuses:
- **proposed**: Initial finding, needs review
- **investigating**: Under active investigation
- **confirmed**: Verified vulnerability
- **rejected**: False positive
- **resolved**: Fixed in code

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

If you have a good idea for improving the analysis, please fork this repository and run comparative benchmarks in the [test environment](https://github.com/muellerberndt/hound-empirical-development).

## License

Apache 2.0 with additional terms:

You may use Hound however you want, except selling it as an online service or as an appliance - that requires written permission from the author.

- See [LICENSE](LICENSE) for details.
