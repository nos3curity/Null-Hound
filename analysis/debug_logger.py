"""
Debug logger for agent LLM interactions.
Captures all prompts and responses in an HTML format for easy debugging.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import html


class DebugLogger:
    """Logs all LLM interactions to an HTML file for debugging."""
    
    def __init__(self, session_id: str, output_dir: Optional[Path] = None):
        """
        Initialize debug logger.
        
        Args:
            session_id: Unique identifier for this session
            output_dir: Directory to save debug logs (defaults to .hound_debug)
        """
        self.session_id = session_id
        self.output_dir = output_dir or Path.home() / ".hound_debug"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create HTML file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"debug_{session_id}_{timestamp}.html"
        
        # Initialize HTML
        self._init_html()
        
        # Track interaction count
        self.interaction_count = 0
        # Track which schemas have already been displayed to avoid repetition
        self._shown_schemas = set()
        # Do not print here; the CLI prints a nice message after finalize()
    
    def _init_html(self):
        """Initialize the HTML file with styling."""
        html_header = """<!DOCTYPE html>
<html>
<head>
    <title>Hound Agent Debug Log</title>
    <style>
        body {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 12px;
            line-height: 1.45;
            font-size: 12px;
        }
        .header {
            background: #2d2d30;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #007acc;
        }
        .header h1 {
            margin: 0;
            color: #4ec9b0;
        }
        .metadata {
            color: #808080;
            margin-top: 10px;
        }
        .interaction {
            background: #2d2d30;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 3px solid #569cd6;
        }
        .interaction-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #404040;
        }
        .interaction-number {
            color: #4ec9b0;
            font-weight: bold;
            font-size: 18px;
        }
        .interaction-time {
            color: #808080;
            font-size: 12px;
        }
        .prompt-section, .response-section {
            margin: 15px 0;
        }
        .section-label {
            color: #569cd6;
            font-weight: bold;
            margin-bottom: 10px;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 1px;
        }
        .system-prompt {
            background: #1e1e1e;
            border: 1px solid #404040;
            border-radius: 4px;
            padding: 10px;
            white-space: pre-wrap;
            color: #ce9178;
            margin-bottom: 10px;
            font-size: 11px;
        }
        .user-prompt {
            background: #1e1e1e;
            border: 1px solid #404040;
            border-radius: 4px;
            padding: 10px;
            white-space: pre-wrap;
            color: #9cdcfe;
            font-size: 11px;
        }
        .response {
            background: #1e1e1e;
            border: 1px solid #404040;
            border-radius: 4px;
            padding: 10px;
            white-space: pre-wrap;
            color: #d4d4d4;
            font-size: 11.5px;
        }
        .schema {
            background: #1e1e1e;
            border: 1px solid #404040;
            border-radius: 4px;
            padding: 10px;
            white-space: pre-wrap;
            color: #b5cea8;
            margin-top: 10px;
            font-size: 10px;
        }
        .error {
            background: #5a1e1e;
            border: 1px solid #f14c4c;
            color: #f48771;
        }
        .tool-call {
            background: #1e3a1e;
            border: 1px solid #4ec9b0;
            color: #4ec9b0;
            margin-top: 10px;
            padding: 10px;
            border-radius: 4px;
        }
        .stats {
            background: #2d2d30;
            border-radius: 8px;
            padding: 15px;
            margin-top: 30px;
            border-left: 3px solid #b5cea8;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }
        .stat-item {
            background: #1e1e1e;
            padding: 10px;
            border-radius: 4px;
        }
        .stat-label {
            color: #808080;
            font-size: 12px;
            text-transform: uppercase;
        }
        .stat-value {
            color: #4ec9b0;
            font-size: 24px;
            font-weight: bold;
        }
        .navigation {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #2d2d30;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .nav-button {
            background: #007acc;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            margin: 0 5px;
        }
        .nav-button:hover {
            background: #005a9e;
        }
        code {
            background: #1e1e1e;
            padding: 2px 6px;
            border-radius: 3px;
            color: #d7ba7d;
            font-size: 12px;
        }
        .duration {
            color: #b5cea8;
            font-size: 12px;
            margin-left: 10px;
        }
    </style>
    <script>
        function scrollToTop() {
            window.scrollTo({top: 0, behavior: 'smooth'});
        }
        function scrollToBottom() {
            window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
        }
    </script>
</head>
<body>
    <div class="header">
        <h1>🔍 Hound Agent Debug Log</h1>
        <div class="metadata">
            <div>Session ID: <code>""" + self.session_id + """</code></div>
            <div>Started: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
        </div>
    </div>
    <div id="interactions">
"""
        
        with open(self.log_file, 'w') as f:
            f.write(html_header)
    
    def log_interaction(
        self,
        system_prompt: str,
        user_prompt: str,
        response: Any,
        schema: Optional[Any] = None,
        duration: Optional[float] = None,
        error: Optional[str] = None,
        tool_calls: Optional[list] = None
    ):
        """
        Log a single LLM interaction.
        
        Args:
            system_prompt: System prompt sent to LLM
            user_prompt: User prompt sent to LLM
            response: Response from LLM (string or parsed object)
            schema: Pydantic schema used for parsing (if any)
            duration: Time taken for the interaction
            error: Error message if interaction failed
            tool_calls: List of tool calls generated
        """
        self.interaction_count += 1
        
        # Format response
        if isinstance(response, str):
            response_html = html.escape(response)
        else:
            try:
                response_html = html.escape(json.dumps(response, indent=2, default=str))
            except:
                response_html = html.escape(str(response))
        
        # Suppress schema display entirely to reduce noise
        schema_html = ""
        
        # Format tool calls
        tool_calls_html = ""
        if tool_calls:
            tool_calls_html = "<div class='section-label'>Generated Tool Calls:</div>"
            for call in tool_calls:
                call_str = f"{call.get('tool_name', 'unknown')}: {json.dumps(call.get('parameters', {}), indent=2)}"
                tool_calls_html += f"<div class='tool-call'>{html.escape(call_str)}</div>"
        
        # Build interaction HTML
        interaction_html = f"""
    <div class="interaction" id="interaction-{self.interaction_count}">
        <div class="interaction-header">
            <span class="interaction-number">Interaction #{self.interaction_count}</span>
            <div>
                <span class="interaction-time">{datetime.now().strftime("%H:%M:%S")}</span>
                {f'<span class="duration">({duration:.2f}s)</span>' if duration else ''}
            </div>
        </div>
        
        <div class="prompt-section">
            <div class="section-label">System Prompt:</div>
            <div class="system-prompt">{html.escape(system_prompt)}</div>
            
            <div class="section-label">User Prompt:</div>
            <div class="user-prompt">{html.escape(user_prompt)}</div>
        </div>
        
        {schema_html}
        
        <div class="response-section">
            <div class="section-label">Response:</div>
            <div class="response {'error' if error else ''}">{response_html if not error else html.escape(error)}</div>
        </div>
        
        {tool_calls_html}
    </div>
"""
        
        # Append to file
        with open(self.log_file, 'a') as f:
            f.write(interaction_html)
    
    def log_event(self, event_type: str, message: str, details: Optional[Dict] = None):
        """
        Log a general event (not an LLM interaction).
        
        Args:
            event_type: Type of event (e.g., "Graph Selection", "Hypothesis Update")
            message: Event message
            details: Optional additional details
        """
        details_html = ""
        if details:
            details_html = f"""
            <div class="section-label">Details:</div>
            <div class="response">{html.escape(json.dumps(details, indent=2, default=str))}</div>
            """
        
        event_html = f"""
    <div class="interaction" style="border-left-color: #b5cea8;">
        <div class="interaction-header">
            <span class="interaction-number" style="color: #b5cea8;">{event_type}</span>
            <span class="interaction-time">{datetime.now().strftime("%H:%M:%S")}</span>
        </div>
        <div style="color: #d4d4d4;">{html.escape(message)}</div>
        {details_html}
    </div>
"""
        
        with open(self.log_file, 'a') as f:
            f.write(event_html)
    
    def finalize(self, summary: Optional[Dict] = None):
        """
        Finalize the debug log with summary statistics.
        
        Args:
            summary: Optional summary statistics to include
        """
        summary_html = ""
        if summary:
            stats_items = ""
            for key, value in summary.items():
                stats_items += f"""
                <div class="stat-item">
                    <div class="stat-label">{key.replace('_', ' ').title()}</div>
                    <div class="stat-value">{value}</div>
                </div>
                """
            
            summary_html = f"""
    <div class="stats">
        <div class="section-label">Session Summary</div>
        <div class="stats-grid">
            {stats_items}
        </div>
    </div>
"""
        
        footer_html = f"""
    {summary_html}
    </div>
    
    <div class="navigation">
        <button class="nav-button" onclick="scrollToTop()">↑ Top</button>
        <button class="nav-button" onclick="scrollToBottom()">↓ Bottom</button>
    </div>
    
    <div class="metadata" style="margin-top: 30px; text-align: center;">
        <div>Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        <div>Total Interactions: {self.interaction_count}</div>
    </div>
</body>
</html>
"""
        
        with open(self.log_file, 'a') as f:
            f.write(footer_html)
        
        return self.log_file
