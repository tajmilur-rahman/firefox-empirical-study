"""
Bug Keyphrase Extraction Script
Uses Mistral 7B Instruct via Ollama to extract keyphrases from bug reports
Fetches data directly from SQL Server or PostgreSQL
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Union
import requests
from tqdm import tqdm

# Import database drivers (import only what's available)
try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

class BugKeyphraseExtractor:
    def __init__(self, config_path: str = "config.json"):
        """Initialize the extractor with configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.ollama_url = self.config.get("ollama_url", "http://localhost:11434")
        self.model_name = self.config.get("model_name", "mistral:7b-instruct")
        self.checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        self.output_file = self.config.get("output_file", "bug_keyphrases.json")

        # Max bugs to process (None = process all)
        self.max_bugs = self.config.get("max_bugs", None)

        # Database type (sqlserver or postgres)
        self.db_type = self.config.get("db_type", "sqlserver").lower()

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Track progress
        self.processed_bugs = self._load_checkpoint()

    def _load_checkpoint(self) -> set:
        """Load processed bug IDs from checkpoint"""
        checkpoint_file = os.path.join(self.checkpoint_dir, "processed_bugs.json")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                print(f"Loaded checkpoint: {len(data['processed'])} bugs already processed")
                return set(data['processed'])
        return set()

    def _save_checkpoint(self, bug_id: int):
        """Save progress checkpoint"""
        self.processed_bugs.add(bug_id)
        checkpoint_file = os.path.join(self.checkpoint_dir, "processed_bugs.json")
        with open(checkpoint_file, 'w') as f:
            json.dump({"processed": list(self.processed_bugs)}, f)

    def connect_to_database(self) -> Union['pyodbc.Connection', 'psycopg2.extensions.connection']:
        """Connect to database (SQL Server or PostgreSQL)"""
        db_config = self.config["database"]

        if self.db_type == "postgres":
            # PostgreSQL connection
            if not PSYCOPG2_AVAILABLE:
                raise ImportError("psycopg2 is not installed. Install with: pip install psycopg2-binary")

            print(f"Connecting to PostgreSQL: {db_config.get('host', 'localhost')}:{db_config.get('port', 5432)}/{db_config['database']}...")

            conn = psycopg2.connect(
                host=db_config.get("host", "localhost"),
                port=db_config.get("port", 5432),
                database=db_config["database"],
                user=db_config["username"],
                password=db_config["password"]
            )
            print("Connected successfully!")
            return conn

        elif self.db_type == "sqlserver":
            # SQL Server connection
            if not PYODBC_AVAILABLE:
                raise ImportError("pyodbc is not installed. Install with: pip install pyodbc")

            # Check TrustServerCertificate setting (auto-detect for Driver 18)
            trust_cert = db_config.get("trust_server_certificate", None)
            if trust_cert is None:
                # Auto-enable for ODBC Driver 18 (required for local dev)
                if "Driver 18" in db_config.get("driver", ""):
                    trust_cert = True
                    print("WARNING: Auto-enabled TrustServerCertificate for ODBC Driver 18")

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

            # Add TrustServerCertificate if needed
            if trust_cert:
                conn_str += "TrustServerCertificate=yes;"

            print(f"Connecting to SQL Server: {db_config['server']}/{db_config['database']}...")
            conn = pyodbc.connect(conn_str)
            print("Connected successfully!")
            return conn

        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}. Must be 'sqlserver' or 'postgres'")

    def _build_queries(self) -> Dict[str, str]:
        """Auto-generate SQL queries based on db_type and schema config"""

        # Check if queries are manually provided in config
        if "queries" in self.config and "fetch_bugs_with_comments" in self.config["queries"]:
            # Use manually provided queries (for custom schemas)
            return self.config["queries"]

        # Auto-generate queries from schema config
        schema = self.config.get("schema", {})
        bugs_table = schema.get("bugs_table", "bugs")
        comments_table = schema.get("comments_table", "comments")

        # Column mappings
        bug_id_col = schema.get("bug_id_column", "id")
        summary_col = schema.get("summary_column", "summary")
        comment_text_col = schema.get("comment_text_column", "text")
        comment_bugid_col = schema.get("comment_bugid_column", "bug_id")
        comment_time_col = schema.get("comment_time_column", "creation_time")

        # Build queries with correct syntax for database type
        if self.db_type == "postgres":
            # PostgreSQL syntax
            placeholder = "%s"
            string_agg = f"STRING_AGG({comments_table}.{comment_text_col}, ' ' ORDER BY {comments_table}.{comment_time_col})"
        else:
            # SQL Server syntax
            placeholder = "?"
            string_agg = f"STRING_AGG({comments_table}.{comment_text_col}, ' ') WITHIN GROUP (ORDER BY {comments_table}.{comment_time_col})"

        queries = {
            "fetch_bugs_with_comments": f"""
                SELECT
                    {bugs_table}.{bug_id_col} AS bug_id,
                    COALESCE({bugs_table}.{summary_col}, '') AS summary,
                    COALESCE({string_agg}, '') AS all_comments
                FROM {bugs_table}
                LEFT JOIN {comments_table} ON {bugs_table}.{bug_id_col} = {comments_table}.{comment_bugid_col}
                GROUP BY {bugs_table}.{bug_id_col}, {bugs_table}.{summary_col}
                ORDER BY {bugs_table}.{bug_id_col}
            """,
            "fetch_bugs": f"SELECT {bug_id_col} AS bug_id, {summary_col} AS summary FROM {bugs_table} ORDER BY {bug_id_col}",
            "fetch_comments": f"SELECT {comment_text_col} AS text FROM {comments_table} WHERE {comment_bugid_col} = {placeholder} ORDER BY {comment_time_col} ASC"
        }

        return queries

    def fetch_bugs(self, conn: Union['pyodbc.Connection', 'psycopg2.extensions.connection']) -> List[Dict[str, Any]]:
        """Fetch all bugs with their comments from database using optimized single query"""

        # Auto-generate or use provided queries
        queries = self._build_queries()

        # Check if using optimized query
        use_optimized = self.config.get("use_optimized_query", True)

        if use_optimized and "fetch_bugs_with_comments" in queries:
            # Use optimized STRING_AGG query (single database round-trip)
            print(f"Fetching bugs with comments (optimized {self.db_type.upper()} query)...")
            query = queries["fetch_bugs_with_comments"]
            cursor = conn.cursor()
            cursor.execute(query)

            bugs = []
            for row in cursor.fetchall():
                bug_id = row[0]

                # Skip if already processed
                if bug_id in self.processed_bugs:
                    continue

                bugs.append({
                    "bug_id": bug_id,
                    "summary": row[1] if len(row) > 1 else "",
                    "all_comments": row[2] if len(row) > 2 else ""
                })

            cursor.close()
            print(f"Fetched {len(bugs)} unprocessed bugs with comments")

            # Apply max_bugs limit if specified
            if self.max_bugs is not None and len(bugs) > self.max_bugs:
                bugs = bugs[:self.max_bugs]
                print(f"WARNING: Limited to {self.max_bugs} bugs (max_bugs setting)")

            return bugs

        else:
            # Fall back to old method (N+1 queries - slower but more compatible)
            query = queries["fetch_bugs"]

            print("Fetching bugs from database...")
            cursor = conn.cursor()
            cursor.execute(query)

            bugs = []
            for row in cursor.fetchall():
                bug_id = row[0]

                # Skip if already processed
                if bug_id in self.processed_bugs:
                    continue

                bugs.append({
                    "bug_id": bug_id,
                    "summary": row[1] if len(row) > 1 else "",
                    "comments": []
                })

            cursor.close()
            print(f" Fetched {len(bugs)} unprocessed bugs")

            # Fetch comments for each bug (WARNING: N+1 query problem)
            print(f"Fetching comments for bugs ({self.db_type.upper()} N+1 queries - consider using optimized query)...")
            comments_query = queries["fetch_comments"]
            cursor = conn.cursor()

            for bug in tqdm(bugs, desc="Loading comments"):
                cursor.execute(comments_query, bug["bug_id"])
                # Fetch and concatenate all comment text
                comment_texts = [row[0] for row in cursor.fetchall() if row[0]]
                bug["all_comments"] = " ".join(comment_texts)

            cursor.close()

            # Apply max_bugs limit if specified
            if self.max_bugs is not None and len(bugs) > self.max_bugs:
                bugs = bugs[:self.max_bugs]
                print(f"WARNING: Limited to {self.max_bugs} bugs (max_bugs setting)")

            return bugs

    def extract_keyphrases_with_ollama(self, bug: Dict[str, Any]) -> List[Dict[str, float]]:
        """Use Mistral 7B via Ollama to extract keyphrases from summary and comments ONLY"""

        # Prepare bug text - ONLY summary and all comments concatenated
        bug_text = f"Summary: {bug['summary']}\n\n"

        if bug.get('all_comments'):
            bug_text += f"Comments:\n{bug['all_comments']}\n"

        # Get keyphrase extraction config
        kp_config = self.config.get("keyphrase_extraction", {})
        min_kp = kp_config.get("min_keyphrases", 5)
        max_kp = kp_config.get("max_keyphrases", 8)
        use_score_ranges = kp_config.get("use_score_ranges", True)

        # Build instructions based on configuration
        if use_score_ranges:
            score_ranges = kp_config.get("score_ranges", {})
            high = score_ranges.get("high", {})
            medium = score_ranges.get("medium", {})
            low = score_ranges.get("low", {})

            score_instructions = f"""2. Assign each a relevance score:
   - {high.get('min', 0.45)}-{high.get('max', 0.55)}: {high.get('description', 'Highly specific technical identifiers')}
   - {medium.get('min', 0.30)}-{medium.get('max', 0.44)}: {medium.get('description', 'Core issue descriptions')}
   - {low.get('min', 0.15)}-{low.get('max', 0.29)}: {low.get('description', 'Supporting context')}"""
        else:
            score_instructions = "2. Assign each a relevance score between 0 and 1 based on how distinctive/central it is to this bug"

        # Create prompt for keyphrase extraction
        prompt = f"""You are a technical keyphrase extraction expert. Extract {min_kp}-{max_kp} keyphrases from this bug report based ONLY on the summary and comments.

Bug Report:
{bug_text}

Instructions:
1. Extract {min_kp}-{max_kp} keyphrases that best represent the core technical issue
{score_instructions}
3. Order from highest to lowest score
4. Output ONLY valid JSON in this exact format, nothing else:

{{"keyphrases": [{{"phrase": "example phrase", "score": 0.48}}, {{"phrase": "another phrase", "score": 0.35}}]}}

JSON output:"""

        # Call Ollama API
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                    }
                },
                timeout=120
            )

            if response.status_code != 200:
                print(f"ERROR: Ollama API error: {response.status_code}")
                return []

            result = response.json()
            generated_text = result.get("response", "").strip()

            # Parse JSON from response
            # Sometimes the model adds extra text, so extract JSON
            json_start = generated_text.find("{")
            json_end = generated_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = generated_text[json_start:json_end]
                parsed = json.loads(json_str)
                return parsed.get("keyphrases", [])
            else:
                print(f"ERROR: Could not find JSON in response for bug {bug['bug_id']}")
                return []

        except requests.exceptions.Timeout:
            print(f"ERROR: Timeout processing bug {bug['bug_id']}")
            return []
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON parse error for bug {bug['bug_id']}: {e}")
            return []
        except Exception as e:
            print(f"ERROR: Error processing bug {bug['bug_id']}: {e}")
            return []

    def process_bugs(self, bugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all bugs and extract keyphrases"""
        results = []

        print(f"\n{'='*60}")
        print(f"Starting keyphrase extraction for {len(bugs)} bugs")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for i, bug in enumerate(tqdm(bugs, desc="Processing bugs")):
            try:
                # Extract keyphrases
                keyphrases = self.extract_keyphrases_with_ollama(bug)

                result = {
                    "bug_id": bug["bug_id"],
                    "tfidf_keyphrases": keyphrases
                }

                results.append(result)

                # Save checkpoint AFTER EVERY bug (not just every 10)
                self._save_checkpoint(bug["bug_id"])

                # Save results every 10 bugs
                if (i + 1) % 10 == 0:
                    self._save_results(results)

                # Progress update every 50 bugs
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = (len(bugs) - i - 1) * avg_time
                    print(f"\n Processed {i+1}/{len(bugs)} bugs")
                    print(f"  Avg time/bug: {avg_time:.2f}s")
                    print(f"  Est. remaining: {remaining/3600:.2f} hours")

            except Exception as e:
                print(f"\nERROR: Error processing bug {bug['bug_id']}: {e}")
                continue

        # Final save
        self._save_results(results)

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f" Processing complete!")
        print(f"  Total bugs: {len(results)}")
        print(f"  Total time: {total_time/3600:.2f} hours")
        print(f"  Avg time/bug: {total_time/len(bugs):.2f}s")
        print(f"{'='*60}\n")

        return results

    def _save_results(self, new_results: List[Dict[str, Any]]):
        """Save results to output file - merges with existing results instead of overwriting"""
        # Load existing results if file exists
        existing_results = []
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupted or empty, start fresh
                existing_results = []

        # Create a dict of existing results by bug_id for fast lookup
        existing_dict = {r["bug_id"]: r for r in existing_results}

        # Merge new results (new results override existing for same bug_id)
        for result in new_results:
            existing_dict[result["bug_id"]] = result

        # Convert back to list and save
        all_results = list(existing_dict.values())

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    def run(self):
        """Main execution flow"""
        print("\n" + "="*60)
        print("Bug Keyphrase Extraction Pipeline")
        print("="*60 + "\n")

        # Connect to database
        conn = self.connect_to_database()

        try:
            # Fetch bugs
            bugs = self.fetch_bugs(conn)

            if not bugs:
                print("No unprocessed bugs found!")
                return

            # Process bugs
            results = self.process_bugs(bugs)

            print(f"\n Results saved to: {self.output_file}")
            print(f" Checkpoints saved in: {self.checkpoint_dir}")

        finally:
            conn.close()
            print("\n Database connection closed")

def main():
    """Entry point"""
    import sys

    config_file = "config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"ERROR: Config file not found: {config_file}")
        print("  Please create config.json (see config.template.json)")
        sys.exit(1)

    extractor = BugKeyphraseExtractor(config_file)
    extractor.run()

if __name__ == "__main__":
    main()
