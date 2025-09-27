#!/usr/bin/env python3
"""
Test script for AI Interface fixes
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_interface import AIInterface

def test_ai_interface():
    """Test the AI interface fixes."""
    print("Testing AI Interface fixes...")

    # Initialize interface
    ai = AIInterface()

    # Test command processing
    test_commands = [
        "show help",
        "show status",
        "setup alpaca",
        "start crypto scalping",
        "show scalping status",
        "stop crypto scalping",
        "show scalping status",
        "start market replay",
        "show status",
        "stop market replay",
        "setup quantconnect",
        "show status"
    ]

    print("\nTesting command processing:")
    for cmd in test_commands:
        print(f"\n--- Testing: '{cmd}' ---")
        try:
            response = ai.process_command(cmd)
            print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")

    # Test status tracking
    print("\n--- Testing Status Tracking ---")
    print(f"System states: {ai.system_states}")

    # Test command repetition prevention
    print("\n--- Testing Command Repetition Prevention ---")
    cmd = "show status"
    print(f"First call to '{cmd}':")
    response1 = ai.process_command(cmd)
    print(f"Response: {response1[:100]}...")

    print(f"Immediate repeat of '{cmd}':")
    response2 = ai.process_command(cmd)
    print(f"Response: {response2}")

    print("\nTest completed!")

if __name__ == "__main__":
    test_ai_interface()