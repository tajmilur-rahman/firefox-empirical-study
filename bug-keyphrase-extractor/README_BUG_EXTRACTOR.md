# Bug Keyphrase Extraction Pipeline

Automated keyphrase extraction from bug reports using Mistral 7B Instruct via Ollama.
Fetches data directly from **SQL Server** or **PostgreSQL** and processes in batches with checkpointing.

**Extraction Source**: Keyphrases are extracted from **SUMMARY + ALL COMMENTS ONLY** for each bug_id.

## Features

- ✅ **Database support**: SQL Server (ODBC) or PostgreSQL
- ✅ Optimized single-query fetch (STRING_AGG) or N+1 fallback
- ✅ Batch processing with automatic checkpointing
- ✅ Resume from interruptions (100% data-safe)
- ✅ Progress tracking with time estimates
- ✅ Uses Mistral 7B Instruct locally (FREE)
- ✅ Extracts 5-8 keyphrases per bug based on summary + comments
- ✅ Handles 250k+ bugs efficiently

## Prerequisites

1. **Python 3.8+** installed
2. **Ollama** running with Mistral 7B Instruct model
3. **Database**: SQL Server *or* PostgreSQL with your bug tracking data
4. **Database Driver**:
   - For SQL Server: ODBC Driver 17/18
   - For PostgreSQL: psycopg2 (auto-installed)

## Installation

### Step 1: Install Required Packages

```bash
pip install -r requirements.txt
```

### Step 2: Install Database Driver

**For SQL Server:**

Install ODBC Driver 17 or 18:
- Windows: https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server

Check installed drivers:
```bash
python -c "import pyodbc; print(pyodbc.drivers())"
```

**For PostgreSQL:**

Uncomment psycopg2 in requirements.txt and install:
```bash
# Uncomment this line in requirements.txt:
# psycopg2-binary>=2.9.0

pip install psycopg2-binary
```

### Step 3: Pull Mistral 7B Instruct Model

```bash
ollama pull mistral:7b-instruct
```

Verify it's running:
```bash
ollama list
```

### Step 4: Configure Database Connection

**Choose One of Two Approaches:**

---

### **Approach 1: Simple (Recommended) - Auto-Generated Queries**

The script automatically generates correct SQL syntax based on `db_type`. You just configure your schema:

```bash
copy config.simple.template.json config.json
```

**Then edit database credentials and table/column names:**

```json
{
  "db_type": "sqlserver",  // or "postgres"
  "database": {
    "driver": "ODBC Driver 18 for SQL Server",
    "server": ".\\SQLEXPRESS",
    "database": "bugbug",
    "trusted_connection": true,
    "trust_server_certificate": true
  },
  "schema": {
    "bugs_table": "bugs",
    "comments_table": "comments",
    "bug_id_column": "id",
    "summary_column": "summary",
    "comment_text_column": "text",
    "comment_bugid_column": "bug_id",
    "comment_time_column": "creation_time"
  }
}
```

**That's it!** The script auto-generates queries with correct syntax (`?` vs `%s`, `WITHIN GROUP` vs not, etc.)

**For PostgreSQL:** Just change `db_type` to `"postgres"` and update the `database` section - everything else stays the same!

```json
{
  "db_type": "postgres",
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "your_database_name",
    "username": "your_username",
    "password": "your_password"
  },
  "schema": {
    // Same as above - no SQL syntax changes needed!
  }
}
```

---

### **Approach 2: Advanced - Manual Queries**

For complex schemas or custom SQL, provide queries directly:

**SQL Server:**
```bash
copy config.template.json config.json
```

**PostgreSQL:**
```bash
copy config.postgres.template.json config.json
```

Then customize the `queries` section with your own SQL (see templates for examples).

---

### Step 5: Verify Setup

Run the setup verification script:

```bash
python verify_setup.py
```

This checks:
- ✓ Python version
- ✓ Required packages installed
- ✓ Database driver available
- ✓ Ollama running with Mistral 7B
- ✓ Database connection works
- ✓ Configuration valid

**Note**: Keyphrases are extracted from `summary` + all `comments.text` (concatenated) ONLY.

## Usage

### Basic Usage

```bash
python bug_keyphrase_extractor.py
```

### With Custom Config

```bash
python bug_keyphrase_extractor.py my_config.json
```

## Output Format

The script generates a JSON file with this structure:

```json
[
  {
    "bug_id": 12345,
    "tfidf_keyphrases": [
      { "phrase": "NullPointerException in loadUserData", "score": 0.52 },
      { "phrase": "user authentication failure", "score": 0.38 },
      { "phrase": "Android 12 platform", "score": 0.22 }
    ]
  }
]
```

## Performance

### Expected Processing Time (250k bugs):

| Hardware | Time per Bug | Total Time (250k) |
|----------|--------------|-------------------|
| RTX 4050 (6GB) | 2-3 seconds | 6-8 days |
| RTX 3060 (12GB) | 1.5-2 seconds | 4-5 days |
| CPU only | 5-8 seconds | 14-18 days |

### Optimization Tips:

1. **Run overnight**: Let it process continuously
2. **Prevent sleep**: Disable power saving on laptop
3. **Monitor progress**: Check the output file periodically
4. **Checkpointing**: Safe to stop/resume anytime

## Checkpointing & Resume

The script automatically saves progress every 10 bugs:
- **Checkpoint file**: `checkpoints/processed_bugs.json`
- **Output file**: Updated continuously

**To resume after interruption:**
Just run the script again - it will skip already processed bugs!

## Troubleshooting

### Issue: "Driver not found"
```bash
# List available drivers
python -c "import pyodbc; print(pyodbc.drivers())"

# Update config.json with exact driver name
```

### Issue: "Connection failed"
```bash
# Test SQL connection (Driver 18)
python -c "import pyodbc; conn = pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER=.\\SQLEXPRESS;Trusted_Connection=yes;TrustServerCertificate=yes;'); print('Connected!')"

# Test SQL connection (Driver 17 - if using older driver)
python -c "import pyodbc; conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost\\SQLEXPRESS;Trusted_Connection=yes;'); print('Connected!')"
```

### Issue: "Ollama not responding"
```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve

# Test model
ollama run mistral:7b-instruct "Hello"
```

### Issue: "Out of memory"
- Close other applications to free up RAM
- Reduce checkpoint frequency (save every 50 bugs instead of 10)
- Use a smaller model if available

## Database Comparison

| Feature | SQL Server | PostgreSQL |
|---------|-----------|------------|
| **Driver** | pyodbc (ODBC) | psycopg2 |
| **Query Placeholder** | `?` | `%s` |
| **STRING_AGG Syntax** | `STRING_AGG(x, ' ') WITHIN GROUP (ORDER BY y)` | `STRING_AGG(x, ' ' ORDER BY y)` |
| **Windows Auth** | `Trusted_Connection=yes` | Not available (use username/password) |
| **Connection** | Connection string | host, port, database, user, password |

## Database Schema Examples

### Example 1: Standard Schema (Bugzilla-style)

```json
{
  "queries": {
    "fetch_bugs": "SELECT bug_id AS bug_id, short_desc AS summary FROM bugs ORDER BY bug_id",
    "fetch_comments": "SELECT thetext AS text FROM longdescs WHERE bug_id = ? ORDER BY bug_when ASC"
  }
}
```

### Example 2: Your Schema (as shown in your SQL)

```json
{
  "queries": {
    "fetch_bugs": "SELECT id AS bug_id, summary FROM bugs ORDER BY id",
    "fetch_comments": "SELECT text FROM comments WHERE bug_id = ? ORDER BY creation_time ASC"
  }
}
```

## Estimated Costs

**Local Processing (Mistral 7B):**
- Cost: $0 (FREE)
- Electricity: ~$3-5 for 7 days continuous
- Total: ~$5

**vs NVIDIA API:**
- Cost: $5-10
- Time: 4-6 hours
- Total: $5-10

## Configuration Options

```json
{
  "db_type": "sqlserver",                     // "sqlserver" or "postgres"
  "ollama_url": "http://localhost:11434",     // Ollama server URL
  "model_name": "mistral:7b-instruct",        // Model to use
  "use_optimized_query": true,                // Use single STRING_AGG query (faster)
  "checkpoint_dir": "checkpoints",            // Where to save progress
  "output_file": "bug_keyphrases_output.json" // Final output
}
```

## Support

For issues or questions:
1. Check the database connection first
2. Verify Ollama is running with the correct model
3. Review the checkpoint files to see progress
4. Check `bug_keyphrases_output.json` for partial results

## License

This script is provided as-is for bug analysis and research purposes.
