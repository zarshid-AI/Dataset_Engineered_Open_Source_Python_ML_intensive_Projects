import requests
import csv
import time
from datetime import datetime, timedelta, timezone
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Dict, Optional
import re
import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    GITHUB_SEARCH_TOKEN, GITHUB_CORE_TOKENS,
    SCRAPER_MIN_STARS, SCRAPER_MIN_FORKS, SCRAPER_MIN_COMMITS,
    SCRAPER_LANGUAGE, SCRAPER_PER_PAGE, SCRAPER_MAX_RESULTS,
    SCRAPER_PARALLEL_WORKERS, SCRAPER_BATCH_SIZE,
    SCRAPER_START_STARS, SCRAPER_END_STARS, SCRAPER_OUTPUT_FILE,
)


# ── Recency filter settings ──────────────────────────────────────────────────
LAST_COMMIT_REFERENCE_DATE = datetime(2026, 2, 28, tzinfo=timezone.utc)
LAST_COMMIT_WINDOW_DAYS = 30


class GitHubRepoScraperCorrected:
    """GitHub scraper with exact CSV column matching"""
    
    def __init__(self, search_token: Optional[str] = None, core_tokens: Optional[List[str]] = None):
        """Initialize the GitHub scraper"""
        self.base_url = "https://api.github.com/search/repositories"
        
        # Search API configuration
        self.search_headers = {'Accept': 'application/vnd.github.v3+json'}
        if search_token:
            self.search_headers['Authorization'] = f'token {search_token}'
        
        # Core API tokens
        self.core_tokens = core_tokens if core_tokens else []
        self.core_headers_list = []
        
        for token in self.core_tokens:
            headers = {'Accept': 'application/vnd.github.v3+json'}
            if token:
                headers['Authorization'] = f'token {token}'
            self.core_headers_list.append(headers)
        
        if not self.core_headers_list:
            self.core_headers_list = [self.search_headers.copy()]
        
        # Thread-safe token rotation
        self.token_index = 0
        self.token_lock = threading.Lock()
        
        # Filter criteria
        self.min_stars = 100
        self.min_forks = 10
        self.min_commits = 100
        self.min_contributors = 10
        self.language = "python"
        self.per_page = 100
        self.max_results_per_query = 1000
        
        # Last-commit cutoff (commit must be on/after this moment)
        reference = LAST_COMMIT_REFERENCE_DATE or datetime.now(timezone.utc)
        self.last_commit_cutoff = reference - timedelta(days=LAST_COMMIT_WINDOW_DAYS)
        
        # HTTP session
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.session.mount('https://', adapter)
        
        # Rate limit cache
        self.rate_limit_cache = {
            'search': {'remaining': 30, 'last_check': 0},
            'core': {'remaining': 5000, 'last_check': 0}
        }
        self.cache_duration = 10
    
    def get_next_core_headers(self) -> Dict[str, str]:
        """Get next core API headers in rotation"""
        with self.token_lock:
            headers = self.core_headers_list[self.token_index]
            self.token_index = (self.token_index + 1) % len(self.core_headers_list)
            return headers.copy()
    
    def check_rate_limit_cached(self, api_type: str = 'search') -> Dict:
        """Check rate limit with caching"""
        current_time = time.time()
        
        if current_time - self.rate_limit_cache[api_type]['last_check'] < self.cache_duration:
            return self.rate_limit_cache[api_type]
        
        headers = self.search_headers if api_type == 'search' else self.core_headers_list[0]
        try:
            response = self.session.get('https://api.github.com/rate_limit', headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                resource_type = 'search' if api_type == 'search' else 'core'
                remaining = data['resources'][resource_type]['remaining']
                reset_time = data['resources'][resource_type]['reset']
                
                self.rate_limit_cache[api_type] = {
                    'remaining': remaining,
                    'reset': reset_time,
                    'last_check': current_time
                }
                return self.rate_limit_cache[api_type]
        except Exception:
            pass
        
        return self.rate_limit_cache[api_type]
    
    def get_commit_count_and_date(self, owner: str, repo: str, headers: Dict) -> tuple:
        """
        Get both commit count and last commit date
        Returns: (commit_count, last_commit_date)
        """
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/commits"
            params = {'per_page': 1, 'page': 1}
            response = self.session.get(url, headers=headers, params=params, timeout=5)
            
            last_commit_date = ''
            commit_count = None
            
            if response.status_code == 200:
                commits_data = response.json()
                
                # Get last commit date from first commit
                if commits_data and len(commits_data) > 0:
                    last_commit_date = commits_data[0]['commit']['committer']['date']
                
                # Get total commit count from Link header
                link_header = response.headers.get('Link', '')
                if 'last' in link_header:
                    match = re.search(r'page=(\d+)>; rel="last"', link_header)
                    if match:
                        commit_count = int(match.group(1))
                else:
                    # Less than 100 commits, get exact count
                    params['per_page'] = 100
                    response2 = self.session.get(url, headers=headers, params=params, timeout=5)
                    if response2.status_code == 200:
                        commit_count = len(response2.json())
            
            elif response.status_code == 409:
                commit_count = 0
                last_commit_date = ''
            
            return (commit_count, last_commit_date)
            
        except Exception:
            return (None, '')
    
    def get_pull_requests(self, owner: str, repo: str, headers: Dict) -> Optional[Dict[str, int]]:
        """
        Get pull request statistics: open, closed (NOT merged) and merged.
        Uses one GraphQL call for exact counts, with a REST fallback.
        Returns None if PR data could not be retrieved.
        """
        # Primary: GraphQL (exact counts in a single request)
        query = """
        query($owner: String!, $name: String!) {
          repository(owner: $owner, name: $name) {
            open:   pullRequests(states: OPEN)   { totalCount }
            closed: pullRequests(states: CLOSED) { totalCount }
            merged: pullRequests(states: MERGED) { totalCount }
          }
        }"""
        try:
            response = self.session.post(
                "https://api.github.com/graphql",
                headers=headers,
                json={'query': query, 'variables': {'owner': owner, 'name': repo}},
                timeout=10,
            )
            if response.status_code == 200:
                repo_node = (response.json().get('data') or {}).get('repository')
                if repo_node:
                    return {
                        'open': repo_node['open']['totalCount'],
                        'closed': repo_node['closed']['totalCount'],  # closed without merge
                        'merged': repo_node['merged']['totalCount'],
                    }
        except Exception:
            pass

        # Fallback: REST (merged/closed counted from at most the 500 latest closed PRs)
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls"

            response = self.session.get(url, headers=headers,
                                        params={'state': 'open', 'per_page': 1}, timeout=5)
            if response.status_code != 200:
                return None
            match = re.search(r'page=(\d+)>; rel="last"', response.headers.get('Link', ''))
            open_prs = int(match.group(1)) if match else len(response.json())

            merged_prs = 0
            closed_prs = 0
            for page in range(1, 6):
                response = self.session.get(
                    url, headers=headers, timeout=5,
                    params={'state': 'closed', 'per_page': 100, 'page': page})
                if response.status_code != 200:
                    return None
                prs = response.json()
                merged_prs += sum(1 for pr in prs if pr.get('merged_at'))
                closed_prs += sum(1 for pr in prs if not pr.get('merged_at'))
                if len(prs) < 100:
                    break

            return {'open': open_prs, 'closed': closed_prs, 'merged': merged_prs}

        except Exception:
            return None
    
    def get_contributors_count(self, owner: str, repo: str, headers: Dict) -> int:
        """Get number of contributors"""
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
            params = {'per_page': 1, 'anon': 'true'}
            response = self.session.get(url, headers=headers, params=params, timeout=5)
            
            if response.status_code == 200:
                link_header = response.headers.get('Link', '')
                if 'last' in link_header:
                    match = re.search(r'page=(\d+)>; rel="last"', link_header)
                    if match:
                        return int(match.group(1))
                return len(response.json())
            
            return 0
            
        except Exception:
            return 0
    
    def has_readme(self, owner: str, repo: str, headers: Dict) -> bool:
        """True if the repository has a README (GitHub returns 404 otherwise)"""
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/readme"
            response = self.session.get(url, headers=headers, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def is_recent_commit(self, last_commit_date: str) -> bool:
        """True if the last commit is on/after the configured cutoff date"""
        if not last_commit_date:
            return False
        try:
            commit_dt = datetime.fromisoformat(last_commit_date.replace('Z', '+00:00'))
        except ValueError:
            return False
        return commit_dt >= self.last_commit_cutoff
    
    def get_repository_details(self, owner: str, repo: str, headers: Dict) -> Optional[Dict]:
        """Get comprehensive repository details"""
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}"
            response = self.session.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            
            return None
            
        except Exception:
            return None
    
    def process_single_repo(self, repo: Dict) -> Optional[Dict]:
        """Process a single repository with exact CSV column structure"""
        headers = self.get_next_core_headers()
        owner = repo['owner']['login']
        repo_name = repo['name']
        
        try:
            # Get detailed repo information
            repo_details = self.get_repository_details(owner, repo_name, headers)
            if not repo_details:
                return None
            
            # Cheap filters first (no extra API calls)
            if repo_details.get('archived', False):
                return None
            if not (repo_details.get('license') or {}).get('key'):
                return None   # no LICENSE detected
            
            # Get commit count and last commit date
            commit_count, last_commit_date = self.get_commit_count_and_date(owner, repo_name, headers)
            if commit_count is None or commit_count < self.min_commits:
                return None
            
            # Last commit must be within the allowed window
            if not self.is_recent_commit(last_commit_date):
                return None
            
            # Must have a README
            if not self.has_readme(owner, repo_name, headers):
                return None
            
            # Get contributors count
            contributors_count = self.get_contributors_count(owner, repo_name, headers)
            if contributors_count < self.min_contributors:
                return None
            
            # Must have at least one open or merged pull request
            pr_stats = self.get_pull_requests(owner, repo_name, headers)
            if pr_stats is None or (pr_stats['open'] + pr_stats['merged']) < 1:
                return None
            
            # Extract license information with proper handling
            license_info = repo_details.get('license')
            
            if license_info is None:
                # No license at all
                license_key = ''
                license_name = ''
                license_spdx_id = ''
                license_url = ''
            else:
                # License exists
                license_key = license_info.get('key', '')
                license_name = license_info.get('name', '')
                license_spdx_id = license_info.get('spdx_id', 'NOASSERTION')
                license_url = license_info.get('url', '')
                
                # Handle special cases
                if license_key == 'other':
                    license_spdx_id = 'NOASSERTION'
                elif not license_spdx_id:
                    license_spdx_id = 'NOASSERTION'
            
            # Calculate size in KB
            size_kb = repo_details.get('size', 0)  # GitHub API already returns size in KB
            
            # Prepare data matching exact CSV structure
            repo_data = {
                # Column 1-6: Basic info
                'name': repo_details['name'],
                'full_name': repo_details['full_name'],
                'owner': owner,
                'description': repo_details.get('description', ''),
                'url': repo_details['html_url'],
                'homepage': repo_details.get('homepage', ''),
                
                # Column 7-10: Metrics
                'stars': repo_details['stargazers_count'],
                'forks': repo_details['forks_count'],
                'watchers': repo_details.get('watchers_count', 0),
                'commits': commit_count,
                
                # Column 11-14: Activity
                'open_issues': repo_details.get('open_issues_count', 0),
                'pull_requests_open': pr_stats['open'],
                'pull_requests_merged': pr_stats['merged'],
                'contributors': contributors_count,
                
                # Column 15-17: Dates
                'created_at': repo_details['created_at'],
                'updated_at': repo_details['updated_at'],
                'last_commit_date': last_commit_date,
                
                # Column 18-20: Language & size
                'language': repo_details.get('language', 'Python'),
                'size_kb': size_kb,
                'default_branch': repo_details.get('default_branch', 'main'),
                
                # Column 21: Topics
                'topics': ', '.join(repo_details.get('topics', [])),
                
                # Column 22-25: License (proper handling)
                'license_key': license_key,
                'license_name': license_name,
                'license_spdx_id': license_spdx_id,
                'license_url': license_url,
                
                # Column 26-33: Features
                'has_issues': repo_details.get('has_issues', False),
                'has_projects': repo_details.get('has_projects', False),
                'has_pages': repo_details.get('has_pages', False),
                'has_wiki': repo_details.get('has_wiki', False),
                'has_downloads': repo_details.get('has_downloads', False),
                'archived': repo_details.get('archived', False),
                'disabled': repo_details.get('disabled', False),
                'visibility': repo_details.get('visibility', 'public'),
                
                # Column 34: Repo language (same as language for Python repos)
                'repo_language': repo_details.get('language', 'Python')
            }

            return repo_data
            
        except Exception as e:
            print(f"Error processing {owner}/{repo_name}: {e}")
            return None
    
    def process_repo_batch(self, repos: List[Dict], max_workers: int = 10) -> List[Dict]:
        """Process multiple repositories in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_single_repo, repo): repo for repo in repos}
            
            for future in as_completed(futures):
                try:
                    repo_data = future.result()
                    if repo_data:
                        results.append(repo_data)
                except Exception as e:
                    print(f"Thread error: {e}")
        
        return results
    
    def search_repositories(self, stars_count: int, page: int = 1) -> Optional[Dict]:
        """Search repositories using GitHub Search API"""
        query = f"language:{self.language} stars:{stars_count} forks:>={self.min_forks}"
        
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': self.per_page,
            'page': page
        }
        
        try:
            response = self.session.get(self.base_url, headers=self.search_headers, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                print("⏳ Search API rate limit hit, waiting 60s...")
                time.sleep(60)
                return self.search_repositories(stars_count, page)
            
            return None
            
        except Exception as e:
            print(f"Search error: {e}")
            return None
    
    def batch_write_csv(self, repos_data: List[Dict], writer, csv_file):
        """Efficiently write multiple repositories to CSV"""
        for repo_data in repos_data:
            writer.writerow(repo_data)
        csv_file.flush()
        os.fsync(csv_file.fileno())
    
    def scrape_repos(self, output_file: str = 'github_python_repos.csv',
                     start_stars: Optional[int] = None, end_stars: Optional[int] = None,
                     parallel_workers: int = 10, batch_size: int = 50):
        """Main scraping function"""
        saved_repos = set()
        total_saved = 0
        skipped_count = 0
        start_time = time.time()
        
        # Load existing repositories
        if os.path.isfile(output_file):
            print(f"📂 Loading existing repos from {output_file}...")
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    saved_repos.add(row['full_name'])
                    total_saved += 1
            print(f"✓ Found {len(saved_repos)} existing repos\n")
        
        current_stars = start_stars if start_stars else self.min_stars
        max_stars = end_stars if end_stars else 500000
        
        print(f"🚀 GITHUB REPOSITORY SCRAPER - CORRECTED VERSION")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  • Language: {self.language}")
        print(f"  • Min Stars: {self.min_stars}")
        print(f"  • Min Forks: {self.min_forks}")
        print(f"  • Min Commits: {self.min_commits}")
        print(f"  • Min Contributors: {self.min_contributors}")
        print(f"  • Parallel Workers: {parallel_workers}")
        print(f"  • Batch Size: {batch_size}")
        print(f"  • Core Tokens: {len(self.core_headers_list)}")
        print(f"  • Star Range: {current_stars} → {max_stars}")
        print(f"  • Filters: archived=False | license + README required | open/merged PR >= 1 | last_commit >= {self.last_commit_cutoff.date()}")
        print(f"{'='*70}\n")
        
        # Exact CSV column structure to match your example
        fieldnames = [
            'name', 'full_name', 'owner', 'description', 'url', 'homepage',
            'stars', 'forks', 'watchers', 'commits', 'open_issues',
            'pull_requests_open', 'pull_requests_merged', 'contributors',
            'created_at', 'updated_at', 'last_commit_date',
            'language', 'size_kb', 'default_branch', 'topics',
            'license_key', 'license_name', 'license_spdx_id', 'license_url',
            'has_issues', 'has_projects', 'has_pages', 'has_wiki', 'has_downloads',
            'archived', 'disabled', 'visibility', 'repo_language'
        ]
        
        file_exists = os.path.isfile(output_file)
        csv_file = open(output_file, 'a', newline='', encoding='utf-8')
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        try:
            while current_stars <= max_stars:
                print(f"\n{'='*70}")
                print(f"⭐ Processing repositories with {current_stars} stars")
                print(f"{'='*70}")
                
                page = 1
                repos_batch = []
                repos_at_this_level = 0
                
                while True:
                    rate_info = self.check_rate_limit_cached('search')
                    if rate_info['remaining'] < 2:
                        wait_time = rate_info['reset'] - time.time() + 5
                        if wait_time > 0:
                            print(f"⏳ Rate limit wait: {wait_time:.0f}s...")
                            time.sleep(wait_time)
                    
                    result = self.search_repositories(current_stars, page)
                    
                    if not result or 'items' not in result:
                        break
                    
                    items = result['items']
                    if not items:
                        break
                    
                    print(f"📄 Page {page}: Found {len(items)} repositories")
                    
                    new_repos = [
                        r for r in items
                        if r['full_name'] not in saved_repos
                        and r['forks_count'] >= self.min_forks
                    ]
                    
                    if new_repos:
                        repos_batch.extend(new_repos)
                    
                    if len(repos_batch) >= batch_size or len(items) < self.per_page:
                        if repos_batch:
                            print(f"🔄 Processing {len(repos_batch)} repos in parallel...")
                            
                            processed_repos = self.process_repo_batch(repos_batch, max_workers=parallel_workers)
                            
                            if processed_repos:
                                self.batch_write_csv(processed_repos, writer, csv_file)
                                print(f"✅ Saved {len(processed_repos)} repos")
                                
                                for repo in processed_repos:
                                    saved_repos.add(repo['full_name'])
                                    total_saved += 1
                                    repos_at_this_level += 1
                            
                            skipped_count += len(repos_batch) - len(processed_repos)
                            repos_batch = []
                    
                    if len(items) < self.per_page or page * self.per_page >= self.max_results_per_query:
                        break
                    
                    page += 1
                
                elapsed = time.time() - start_time
                rate = total_saved / elapsed if elapsed > 0 else 0
                
                print(f"\n📊 Statistics:")
                print(f"  • Repos at {current_stars}★: {repos_at_this_level}")
                print(f"  • Total saved: {total_saved}")
                print(f"  • Skipped: {skipped_count}")
                print(f"  • Rate: {rate:.1f} repos/sec ({rate*60:.0f} repos/min)")
                print(f"  • Elapsed: {elapsed/60:.1f} minutes")
                
                current_stars += 1
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user. Progress saved.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            csv_file.close()
            
            elapsed = time.time() - start_time
            print(f"\n{'='*70}")
            print(f"✅ SCRAPING COMPLETED")
            print(f"{'='*70}")
            print(f"  Total repositories saved: {total_saved}")
            print(f"  Total time: {elapsed/60:.1f} minutes")
            print(f"  Average rate: {total_saved/(elapsed/60):.0f} repos/min")
            print(f"  Output file: {output_file}")
            print(f"{'='*70}\n")


def main():
    """Main execution function"""
    print("🔧 GitHub Scraper - Corrected Version")
    print("   • Config loaded from config.py / .env")
    print("   • Exact CSV column structure")
    print("   • Proper license handling (NOASSERTION vs empty)")
    print("   • Last commit date included")
    print("   • Filters: archived=False | license + README | open/merged PR | recent last commit")
    print()

    scraper = GitHubRepoScraperCorrected(
        search_token=GITHUB_SEARCH_TOKEN,
        core_tokens=GITHUB_CORE_TOKENS,
    )

    # Configure filters from config
    scraper.min_stars        = SCRAPER_MIN_STARS
    scraper.min_forks        = SCRAPER_MIN_FORKS
    scraper.min_commits      = SCRAPER_MIN_COMMITS
    scraper.min_contributors = 10
    scraper.language         = SCRAPER_LANGUAGE

    # Start scraping
    scraper.scrape_repos(
        output_file=SCRAPER_OUTPUT_FILE,
        start_stars=SCRAPER_START_STARS,
        end_stars=SCRAPER_END_STARS,
        parallel_workers=SCRAPER_PARALLEL_WORKERS,
        batch_size=SCRAPER_BATCH_SIZE,
    )


if __name__ == "__main__":
    main()