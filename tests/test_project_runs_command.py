"""Unit tests for project runs command."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from commands.project import _list_runs, _show_run_details


class TestProjectRunsCommand(unittest.TestCase):
    """Test project runs command functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test runs
        self.temp_dir = tempfile.mkdtemp()
        self.runs_dir = Path(self.temp_dir)
        
        # Create sample run files
        self.sample_runs = [
            {
                'run_id': 'run_123',
                'command_args': ['hound', 'agent', 'test', '--iterations', '5'],
                'start_time': '2025-08-26T10:00:00',
                'end_time': '2025-08-26T10:05:00',
                'runtime_seconds': 300.5,
                'status': 'completed',
                'token_usage': {
                    'total_usage': {
                        'input_tokens': 5000,
                        'output_tokens': 2500,
                        'total_tokens': 7500,
                        'call_count': 10
                    },
                    'by_model': {
                        'openai:gpt-4': {
                            'input_tokens': 3000,
                            'output_tokens': 1500,
                            'total_tokens': 4500,
                            'call_count': 6
                        },
                        'openai:gpt-5': {
                            'input_tokens': 2000,
                            'output_tokens': 1000,
                            'total_tokens': 3000,
                            'call_count': 4
                        }
                    }
                },
                'investigations': [
                    {
                        'goal': 'Find security vulnerabilities',
                        'priority': 1,
                        'category': 'security',
                        'iterations_completed': 5,
                        'hypotheses': {'total': 3, 'high': 1, 'medium': 2}
                    }
                ],
                'errors': []
            },
            {
                'run_id': 'run_456',
                'command_args': ['hound', 'agent', 'test2'],
                'start_time': '2025-08-26T11:00:00',
                'end_time': None,
                'runtime_seconds': 150.0,
                'status': 'interrupted',
                'token_usage': {
                    'total_usage': {
                        'input_tokens': 2000,
                        'output_tokens': 1000,
                        'total_tokens': 3000,
                        'call_count': 5
                    }
                },
                'investigations': [],
                'errors': [
                    {
                        'timestamp': '2025-08-26T11:02:30',
                        'error': 'User interrupted'
                    }
                ]
            }
        ]
        
        # Write sample run files
        for run_data in self.sample_runs:
            run_file = self.runs_dir / f"{run_data['run_id']}.json"
            with open(run_file, 'w') as f:
                json.dump(run_data, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('click.echo')
    @patch('commands.project.console')
    def test_list_runs_json_output(self, mock_console, mock_echo):
        """Test listing runs with JSON output."""
        _list_runs(self.runs_dir, output_json=True)
        
        # Check that JSON was output
        self.assertTrue(mock_echo.called)
        json_output = mock_echo.call_args[0][0]
        data = json.loads(json_output)
        
        # Verify data structure
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['run_id'], 'run_123')
        self.assertEqual(data[0]['status'], 'completed')
        self.assertEqual(data[0]['total_tokens'], 7500)
        self.assertEqual(data[1]['run_id'], 'run_456')
        self.assertEqual(data[1]['status'], 'interrupted')
    
    @patch('commands.project.console')
    def test_list_runs_table_output(self, mock_console):
        """Test listing runs with table output."""
        _list_runs(self.runs_dir, output_json=False)
        
        # Check that console.print was called with a table
        self.assertTrue(mock_console.print.called)
        
        # Get the table from the call
        for call in mock_console.print.call_args_list:
            args = call[0]
            if args and hasattr(args[0], 'add_column'):
                # Found the table
                table = args[0]
                self.assertIsNotNone(table)
                break
        else:
            self.fail("No table was printed")
    
    @patch('click.echo')
    @patch('commands.project.console')
    def test_show_run_details_json(self, mock_console, mock_echo):
        """Test showing run details with JSON output."""
        _show_run_details(self.runs_dir, 'run_123', output_json=True)
        
        # Check that JSON was output
        self.assertTrue(mock_echo.called)
        json_output = mock_echo.call_args[0][0]
        data = json.loads(json_output)
        
        # Verify it's the correct run
        self.assertEqual(data['run_id'], 'run_123')
        self.assertEqual(data['status'], 'completed')
    
    @patch('commands.project.console')
    def test_show_run_details_formatted(self, mock_console):
        """Test showing run details with formatted output."""
        _show_run_details(self.runs_dir, 'run_123', output_json=False)
        
        # Check that console.print was called multiple times
        self.assertTrue(mock_console.print.called)
        
        # Check that key information was printed
        printed_text = ' '.join(str(call[0][0]) for call in mock_console.print.call_args_list if call[0])
        
        # Verify key information is in output
        self.assertIn('run_123', printed_text)
        self.assertIn('completed', printed_text)
        self.assertIn('5000', printed_text)  # input tokens
        self.assertIn('2500', printed_text)  # output tokens
    
    @patch('commands.project.console')
    def test_show_run_details_not_found(self, mock_console):
        """Test showing details for non-existent run."""
        _show_run_details(self.runs_dir, 'run_999', output_json=False)
        
        # Check that error message was printed
        self.assertTrue(mock_console.print.called)
        printed_text = ' '.join(str(call[0][0]) for call in mock_console.print.call_args_list if call[0])
        self.assertIn("not found", printed_text.lower())
    
    @patch('commands.project.console')
    def test_empty_runs_directory(self, mock_console):
        """Test listing runs from empty directory."""
        empty_dir = Path(tempfile.mkdtemp())
        try:
            _list_runs(empty_dir, output_json=False)
            
            # Check that appropriate message was printed
            self.assertTrue(mock_console.print.called)
            printed_text = ' '.join(str(call[0][0]) for call in mock_console.print.call_args_list if call[0])
            self.assertIn("no", printed_text.lower())
        finally:
            import shutil
            shutil.rmtree(empty_dir)
    
    def test_malformed_run_file(self):
        """Test handling of malformed run file."""
        # Create a malformed JSON file
        bad_file = self.runs_dir / "run_bad.json"
        with open(bad_file, 'w') as f:
            f.write("not valid json{")
        
        # Should not crash, should skip the bad file
        with patch('commands.project.console') as mock_console:
            _list_runs(self.runs_dir, output_json=False)
            
            # Check warning was printed
            warning_found = False
            for call in mock_console.print.call_args_list:
                if call[0] and "Warning" in str(call[0][0]):
                    warning_found = True
                    break
            self.assertTrue(warning_found)


if __name__ == '__main__':
    unittest.main()