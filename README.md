# Null-Hound

**Autonomous AI security auditor with adaptive knowledge graphs**

Custom fork of [Hound](https://github.com/muellerberndt/hound) with a preset system and a focus on high-impact vulnerabilities

---

## Quick Start

### Install

```bash
git clone https://github.com/yourusername/Null-Hound.git
cd Null-Hound
pip install -r requirements.txt
```

### Configure

```bash
# Set up API keys
export GOOGLE_API_KEY=your_gemini_key_here
```

### Usage

```bash
# 1. Create project with preset (solidity, rust, or default)
./hound.py project create myaudit /path/to/code solidity

# 2. Generate smart file filter
./hound.py filter myaudit

# 3. Build knowledge graphs (uses preset configuration)
./hound.py graph build myaudit

# Or specify number of additional auto-generated graphs
./hound.py graph build myaudit --graphs 5

# Or disable additional graphs
./hound.py graph build myaudit --graphs 0

# 4. Run security audit (Phase 1: Sweep)
./hound.py agent audit myaudit --mode sweep

# 5. Deep analysis (Phase 2: Intuition)
./hound.py agent audit myaudit --mode intuition --time-limit 300

# 6. Review findings
./hound.py finalize myaudit

# 7. Generate report
./hound.py report myaudit
```

---

## Features

- **Preset System** - Optimized configs for Solidity, Rust, or general code
- **Gemini-Powered Filtering** - Intelligently selects security-relevant files
- **Dynamic Knowledge Graphs** - Agent-designed, multi-aspect code understanding
- **Strategic Planning** - Balances broad coverage with deep investigation
- **Multi-Provider LLM Support** - OpenAI, Anthropic, Gemini, DeepSeek, xAI

---

## Documentation

- **[Full Documentation](docs/FULL_DOCUMENTATION.md)** - Complete guide with all features
- **[Technical Details](docs/tech.md)** - Architecture and implementation
- **[Internal Docs](CLAUDE.md)** - For AI assistants working on this codebase

---

## Links

- [Original Hound](https://github.com/muellerberndt/hound)
- [Research Paper](https://zenodo.org/records/17221190)
- [Blog Post](https://muellerberndt.medium.com/unleashing-the-hound-how-ai-agents-find-deep-logic-bugs-in-any-codebase-64c2110e3a6f)

---

## License

Apache 2.0