"""
Setup Verification Script
Checks all prerequisites before running the bug keyphrase extractor
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (need 3.8+)")
        return False

def check_packages():
    """Check required Python packages"""
    print("\nChecking required packages...")
    packages = {
        'pyodbc': 'pyodbc',
        'requests': 'requests',
        'tqdm': 'tqdm'
    }

    all_installed = True
    for package, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - Run: pip install {package}")
            all_installed = False

    return all_installed

def check_odbc_drivers():
    """Check ODBC drivers"""
    print("\nChecking ODBC drivers...")
    try:
        import pyodbc
        drivers = pyodbc.drivers()
        if drivers:
            print("  Available drivers:")
            for driver in drivers:
                print(f"    - {driver}")
            return True
        else:
            print("  ✗ No ODBC drivers found")
            print("    Install from: https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def check_ollama():
    """Check if Ollama is running"""
    print("\nChecking Ollama...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("  ✓ Ollama is running")
            print("  Available models:")
            for model in models:
                print(f"    - {model['name']}")

            # Check for mistral
            mistral_found = any('mistral' in model['name'] for model in models)
            if mistral_found:
                print("  ✓ Mistral model found")
                return True
            else:
                print("  ⚠ Mistral 7B not found - Run: ollama pull mistral:7b-instruct")
                return False
        else:
            print("  ✗ Ollama not responding")
            return False
    except Exception as e:
        print(f"  ✗ Ollama not running: {e}")
        print("    Start with: ollama serve")
        return False

def check_config():
    """Check if config.json exists"""
    print("\nChecking configuration...")
    if os.path.exists("config.json"):
        print("  ✓ config.json found")
        try:
            import json
            with open("config.json", 'r') as f:
                config = json.load(f)

            # Check required fields
            required = ['database', 'queries', 'ollama_url', 'model_name']
            missing = [field for field in required if field not in config]

            if missing:
                print(f"  ⚠ Missing fields: {', '.join(missing)}")
                return False

            print("  ✓ Configuration valid")
            return True
        except json.JSONDecodeError:
            print("  ✗ config.json is not valid JSON")
            return False
    else:
        print("  ✗ config.json not found")
        print("    Copy config.template.json to config.json and edit it")
        return False

def test_database_connection():
    """Test database connection"""
    print("\nTesting database connection...")
    if not os.path.exists("config.json"):
        print("  ⚠ Skipping (no config.json)")
        return False

    try:
        import json
        import pyodbc

        with open("config.json", 'r') as f:
            config = json.load(f)

        db_config = config['database']

        # Build connection string
        if db_config.get("trusted_connection", False):
            conn_str = (
                f"DRIVER={{{db_config['driver']}}};"
                f"SERVER={db_config['server']};"
                f"DATABASE={db_config['database']};"
                f"Trusted_Connection=yes;"
            )
        else:
            conn_str = (
                f"DRIVER={{{db_config['driver']}}};"
                f"SERVER={db_config['server']};"
                f"DATABASE={db_config['database']};"
                f"UID={db_config['username']};"
                f"PWD={db_config['password']};"
            )

        conn = pyodbc.connect(conn_str, timeout=10)
        conn.close()
        print("  ✓ Database connection successful")
        return True

    except Exception as e:
        print(f"  ✗ Database connection failed: {e}")
        print("    Check your config.json database settings")
        return False

def main():
    """Run all checks"""
    print("="*60)
    print("Bug Keyphrase Extractor - Setup Verification")
    print("="*60 + "\n")

    checks = [
        ("Python version", check_python_version()),
        ("Python packages", check_packages()),
        ("ODBC drivers", check_odbc_drivers()),
        ("Ollama service", check_ollama()),
        ("Configuration", check_config()),
        ("Database connection", test_database_connection()),
    ]

    print("\n" + "="*60)
    print("Summary:")
    print("="*60)

    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")

    print("\n" + "="*60)
    if passed == total:
        print("✓ ALL CHECKS PASSED! Ready to run the extractor.")
        print("\nRun: python bug_keyphrase_extractor.py")
    else:
        print(f"⚠ {total - passed} check(s) failed. Fix the issues above.")
        print("\nNext steps:")
        if not checks[1][1]:  # packages
            print("  1. pip install -r requirements.txt")
        if not checks[3][1]:  # ollama
            print("  2. ollama pull mistral:7b-instruct")
        if not checks[4][1]:  # config
            print("  3. Copy config.template.json to config.json and edit it")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
