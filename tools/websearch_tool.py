#!/usr/bin/env python3
"""
Web Search Tool using Perplexity API.

This module provides functions to perform web searches using Perplexity's Sonar API
and return natural language responses with citations.
"""

import os
import json
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def websearch(query: str, model: str = "sonar-pro") -> str:
    """
    Perform a web search using Perplexity's Sonar API.
    
    Args:
        query: The search query to execute
        model: The Perplexity model to use (default: "sonar-pro")
        
    Returns:
        Natural language response from the search model as a string
        
    Raises:
        ValueError: If API key is not configured or API call fails
        requests.RequestException: If there's a network error
    """
    # Get API key from environment
    api_key = os.getenv('PERPLEXITY_API_KEY')
    
    if not api_key:
        raise ValueError(
            "PERPLEXITY_API_KEY environment variable not set. "
            "Please set it with your Perplexity API key."
        )
    
    # API endpoint
    url = "https://api.perplexity.ai/chat/completions"
    
    # Headers
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Request payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Be precise and concise. Provide accurate information with proper citations when available."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    try:
        # Make the API request
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the response
        data = response.json()
        
        # Extract the content from the response
        if 'choices' in data and len(data['choices']) > 0:
            message = data['choices'][0].get('message', {})
            content = message.get('content', '')
            
            if content:
                return content
            else:
                return "No content returned from search."
        else:
            return "No search results found."
            
    except requests.RequestException as e:
        raise requests.RequestException(f"Network error during search: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from API: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error performing web search: {str(e)}")


def websearch_with_metadata(query: str, model: str = "sonar-pro") -> Dict[str, Any]:
    """
    Perform a web search and return response with metadata.
    
    Args:
        query: The search query to execute
        model: The Perplexity model to use (default: "sonar-pro")
        
    Returns:
        Dictionary containing response content and metadata
        
    Raises:
        ValueError: If API key is not configured or API call fails
        requests.RequestException: If there's a network error
    """
    # Get API key from environment
    api_key = os.getenv('PERPLEXITY_API_KEY')
    
    if not api_key:
        raise ValueError(
            "PERPLEXITY_API_KEY environment variable not set. "
            "Please set it with your Perplexity API key."
        )
    
    # API endpoint
    url = "https://api.perplexity.ai/chat/completions"
    
    # Headers
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Request payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Be precise and concise. Provide accurate information with proper citations when available."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    try:
        # Make the API request
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        
        # Extract the content and metadata
        result = {
            'query': query,
            'model': model,
            'content': '',
            'usage': {},
            'citations': []
        }
        
        if 'choices' in data and len(data['choices']) > 0:
            message = data['choices'][0].get('message', {})
            result['content'] = message.get('content', 'No content returned from search.')
            
            # Extract usage information if available
            if 'usage' in data:
                result['usage'] = data['usage']
            
            # Extract citations if available
            if 'citations' in message:
                result['citations'] = message['citations']
        else:
            result['content'] = "No search results found."
        
        return result
        
    except requests.RequestException as e:
        raise requests.RequestException(f"Network error during search: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from API: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error performing web search: {str(e)}")


def test_websearch_connection() -> bool:
    """
    Test if the websearch tool can connect to the Perplexity API.
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        # Simple test query
        response = websearch("What is the current year?")
        return len(response) > 0
    except Exception as e:
        print(f"Websearch connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the websearch tool
    import sys
    
    # Check if API key is configured
    if not os.getenv('PERPLEXITY_API_KEY'):
        print("Error: PERPLEXITY_API_KEY environment variable not set")
        print("Please set it with your Perplexity API key from https://docs.perplexity.ai/guides/getting-started")
        sys.exit(1)
    
    # Test query
    test_query = "Latest advances in protein stability engineering 2024"
    
    try:
        print(f"Testing websearch with query: '{test_query}'")
        print("=" * 60)
        
        # Test basic search
        result = websearch(test_query)
        print("Search Result:")
        print(result)
        
        print("\n" + "=" * 60)
        print("Testing detailed search...")
        
        # Test detailed search
        detailed_result = websearch_with_metadata(test_query)
        print(f"Query: {detailed_result['query']}")
        print(f"Model: {detailed_result['model']}")
        print(f"Content: {detailed_result['content']}")
        
        if detailed_result['usage']:
            print(f"Usage: {detailed_result['usage']}")
        
        if detailed_result['citations']:
            print(f"Citations: {detailed_result['citations']}")
        
        print("\n✓ Websearch tool is working correctly!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 