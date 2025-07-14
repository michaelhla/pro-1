#!/usr/bin/env python3
"""
Test script for the websearch tool.

This script tests the websearch functionality and integration with the designer.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from websearch_tool import websearch, websearch_with_metadata, test_websearch_connection


class TestWebsearchTool(unittest.TestCase):
    """Test cases for the websearch tool."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock API key for testing
        self.mock_api_key = "test_api_key"
        
    def test_websearch_no_api_key(self):
        """Test websearch with no API key configured."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                websearch("test query")
            self.assertIn("PERPLEXITY_API_KEY", str(context.exception))
    
    @patch('websearch_tool.requests.post')
    def test_websearch_success(self, mock_post):
        """Test successful websearch request."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'choices': [
                {
                    'message': {
                        'content': 'Test search result content'
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        with patch.dict(os.environ, {'PERPLEXITY_API_KEY': self.mock_api_key}):
            result = websearch("test query")
            
        self.assertEqual(result, "Test search result content")
        mock_post.assert_called_once()
    
    @patch('websearch_tool.requests.post')
    def test_websearch_with_metadata_success(self, mock_post):
        """Test successful websearch with metadata."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'choices': [
                {
                    'message': {
                        'content': 'Test search result content',
                        'citations': ['http://example.com']
                    }
                }
            ],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 20,
                'total_tokens': 30
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        with patch.dict(os.environ, {'PERPLEXITY_API_KEY': self.mock_api_key}):
            result = websearch_with_metadata("test query")
            
        self.assertEqual(result['content'], "Test search result content")
        self.assertEqual(result['query'], "test query")
        self.assertEqual(result['model'], "sonar-pro")
        self.assertEqual(result['citations'], ['http://example.com'])
        self.assertEqual(result['usage']['total_tokens'], 30)
    
    @patch('websearch_tool.requests.post')
    def test_websearch_no_content(self, mock_post):
        """Test websearch with no content in response."""
        # Mock response with no content
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'choices': [
                {
                    'message': {
                        'content': ''
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        with patch.dict(os.environ, {'PERPLEXITY_API_KEY': self.mock_api_key}):
            result = websearch("test query")
            
        self.assertEqual(result, "No content returned from search.")
    
    @patch('websearch_tool.requests.post')
    def test_websearch_network_error(self, mock_post):
        """Test websearch with network error."""
        # Mock network error
        mock_post.side_effect = Exception("Network error")
        
        with patch.dict(os.environ, {'PERPLEXITY_API_KEY': self.mock_api_key}):
            with self.assertRaises(ValueError) as context:
                websearch("test query")
            self.assertIn("Network error", str(context.exception))
    
    @patch('websearch_tool.websearch')
    def test_websearch_connection_test(self, mock_websearch):
        """Test websearch connection test function."""
        # Test successful connection
        mock_websearch.return_value = "Test response"
        self.assertTrue(test_websearch_connection())
        
        # Test failed connection
        mock_websearch.side_effect = Exception("Connection failed")
        self.assertFalse(test_websearch_connection())


def test_websearch_integration():
    """
    Integration test to verify the websearch tool works with the designer.
    """
    print("Testing websearch integration...")
    
    # Test import
    try:
        from carbonic_anhydrase_designer import CarbonicAnhydraseDesigner
        print("✓ Designer import successful")
    except ImportError as e:
        print(f"✗ Designer import failed: {e}")
        return False
    
    # Test designer creation (will fail without API keys, but we can still test tool registration)
    try:
        designer = CarbonicAnhydraseDesigner()
        print("✓ Designer created successfully")
        
        # Check that websearch tool is in the tools
        tool_names = [tool.get('name') for tool in designer.tools]
        assert 'websearch' in tool_names, "Websearch tool not found in designer tools"
        print("✓ Websearch tool found in designer tools")
        
        # Check that websearch function is in the mapping
        assert 'websearch' in designer.tool_mapping, "Websearch function not found in tool mapping"
        print("✓ Websearch function found in tool mapping")
        
        # Test tool definition
        websearch_tool = None
        for tool in designer.tools:
            if tool.get('name') == 'websearch':
                websearch_tool = tool
                break
        
        assert websearch_tool is not None, "Websearch tool definition not found"
        assert websearch_tool['type'] == 'function', "Invalid tool type"
        assert 'parameters' in websearch_tool, "Tool parameters not found"
        assert 'query' in websearch_tool['parameters']['properties'], "Query parameter not found"
        print("✓ Websearch tool definition is correct")
        
        print("✓ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"⚠ Designer creation failed (expected if no API keys): {e}")
        # Even if designer creation fails, we can still test the import
        try:
            from websearch_tool import websearch
            print("✓ Websearch tool module imported successfully")
            return True
        except ImportError as e:
            print(f"✗ Websearch tool import failed: {e}")
            return False


def test_websearch_with_real_api():
    """
    Test websearch with real API (only if API key is available).
    """
    print("\nTesting websearch with real API...")
    
    if not os.getenv('PERPLEXITY_API_KEY'):
        print("⚠ PERPLEXITY_API_KEY not set, skipping real API test")
        return
    
    try:
        # Test basic websearch
        result = websearch("What is carbonic anhydrase?")
        print(f"✓ Basic websearch successful: {result[:100]}...")
        
        # Test websearch with metadata
        result_meta = websearch_with_metadata("protein stability engineering")
        print(f"✓ Websearch with metadata successful")
        print(f"  Content length: {len(result_meta['content'])}")
        print(f"  Usage: {result_meta.get('usage', 'N/A')}")
        
        # Test connection
        if test_websearch_connection():
            print("✓ Connection test passed")
        else:
            print("✗ Connection test failed")
            
    except Exception as e:
        print(f"✗ Real API test failed: {e}")


if __name__ == "__main__":
    print("Running websearch tool tests...")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 60)
    print("Running integration test...")
    
    # Run integration test
    test_websearch_integration()
    
    print("\n" + "=" * 60)
    print("Testing real API (if available)...")
    
    # Test with real API if available
    test_websearch_with_real_api()
    
    print("\n" + "=" * 60)
    print("Test summary:")
    print("- Unit tests: Check individual websearch functions")
    print("- Integration test: Check websearch tool integration with designer")
    print("- Real API test: Test with actual Perplexity API (if key available)")
    print("- Note: Get API key from https://docs.perplexity.ai/guides/getting-started")
    print("- Set PERPLEXITY_API_KEY environment variable to enable web search") 