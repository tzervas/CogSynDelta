"""Documentation ingestion pipeline for fetching and processing dependency docs.

This module handles fetching documentation from PyPI, GitHub releases,
and official documentation sites, then preparing it for the RAG system.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from llama_index.core import Document


def _checked_urlopen(url_or_req, *, timeout: int = 10):
    """urlopen restricted to http/https.

    Bare urlopen honours file:, ftp: and custom schemes, so a URL that reaches this
    from config or an API response can read local files. Validate before opening.
    """
    target = url_or_req if isinstance(url_or_req, str) else url_or_req.full_url
    scheme = urllib.parse.urlparse(target).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"refusing non-http(s) URL scheme {scheme!r}: {target}")
    # Scheme is validated immediately above; this IS the guard.
    return urllib.request.urlopen(url_or_req, timeout=timeout)  # nosec B310


class DocumentIngestionPipeline:
    """Pipeline for ingesting documentation from various sources."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize the ingestion pipeline.

        Args:
            cache_dir: Directory to cache fetched documentation
        """
        self.cache_dir = cache_dir or Path("./data/doc_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_pypi_info(
        self, package_name: str, version: str | None = None
    ) -> tuple[dict[str, Any], str]:
        """Fetch package information from PyPI.

        Args:
            package_name: Name of the package
            version: Specific version, or None for latest

        Returns:
            Tuple of (package_info_dict, version_string)
        """
        url = f"https://pypi.org/pypi/{package_name}"
        if version:
            url += f"/{version}"
        url += "/json"

        try:
            with _checked_urlopen(url, timeout=10) as response:
                data = json.loads(response.read())
                actual_version = data["info"]["version"]
                return data, actual_version
        except Exception as e:
            print(f"Error fetching PyPI info for {package_name}: {e}")
            return {}, ""

    def fetch_github_release_notes(self, repo_owner: str, repo_name: str, tag: str) -> str | None:
        """Fetch release notes from GitHub.

        Args:
            repo_owner: GitHub repository owner
            repo_name: Repository name
            tag: Release tag (e.g., 'v2.9.1')

        Returns:
            Release notes text or None
        """
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/tags/{tag}"

        try:
            with _checked_urlopen(url, timeout=10) as response:
                data = json.loads(response.read())
                return data.get("body", "")
        except Exception as e:
            print(f"Error fetching GitHub release for {repo_owner}/{repo_name}/{tag}: {e}")
            return None

    def fetch_url_content(self, url: str) -> str | None:
        """Fetch content from a URL.

        Args:
            url: URL to fetch

        Returns:
            Content as string or None
        """
        try:
            with _checked_urlopen(url, timeout=30) as response:
                content = response.read()
                # Try to decode as UTF-8
                return content.decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"Error fetching URL {url}: {e}")
            return None

    def create_documents_from_pypi(
        self, package_name: str, version: str | None = None
    ) -> list[Document]:
        """Create LlamaIndex documents from PyPI package information.

        Args:
            package_name: Name of the package
            version: Specific version, or None for latest

        Returns:
            List of Document objects
        """
        package_info, actual_version = self.fetch_pypi_info(package_name, version)
        if not package_info:
            return []

        documents = []
        info = package_info.get("info", {})

        # Main package description
        description = info.get("description", "")
        if description:
            doc = Document(
                text=description,
                metadata={
                    "title": f"{package_name} v{actual_version} - PyPI Description",
                    "source": "pypi",
                    "url": f"https://pypi.org/project/{package_name}/{actual_version}/",
                },
            )
            documents.append(doc)

        # Package metadata summary
        summary_parts = [
            f"Package: {package_name}",
            f"Version: {actual_version}",
            f"Summary: {info.get('summary', 'N/A')}",
            f"Author: {info.get('author', 'N/A')}",
            f"License: {info.get('license', 'N/A')}",
            f"Homepage: {info.get('home_page', 'N/A')}",
            f"Project URLs: {json.dumps(info.get('project_urls', {}), indent=2)}",
        ]

        # Add requirements
        requires_dist = info.get("requires_dist", [])
        if requires_dist:
            summary_parts.append("\nDependencies:")
            for req in requires_dist[:20]:  # Limit to first 20
                summary_parts.append(f"  - {req}")

        summary_text = "\n".join(summary_parts)
        doc = Document(
            text=summary_text,
            metadata={
                "title": f"{package_name} v{actual_version} - Package Metadata",
                "source": "pypi",
                "url": f"https://pypi.org/project/{package_name}/{actual_version}/",
            },
        )
        documents.append(doc)

        return documents

    def create_documents_from_github_release(
        self, repo_owner: str, repo_name: str, tag: str, package_name: str
    ) -> list[Document]:
        """Create documents from GitHub release notes.

        Args:
            repo_owner: GitHub repository owner
            repo_name: Repository name
            tag: Release tag
            package_name: Package name for metadata

        Returns:
            List of Document objects
        """
        release_notes = self.fetch_github_release_notes(repo_owner, repo_name, tag)
        if not release_notes:
            return []

        doc = Document(
            text=release_notes,
            metadata={
                "title": f"{package_name} {tag} - GitHub Release Notes",
                "source": "github",
                "url": f"https://github.com/{repo_owner}/{repo_name}/releases/tag/{tag}",
            },
        )

        return [doc]

    def create_documents_from_url(
        self, url: str, title: str, package_name: str, chunk_size: int = 10000
    ) -> list[Document]:
        """Create documents from a documentation URL.

        Args:
            url: Documentation URL
            title: Title for the document
            package_name: Package name for metadata
            chunk_size: Maximum size per chunk

        Returns:
            List of Document objects
        """
        content = self.fetch_url_content(url)
        if not content:
            return []

        # Clean up HTML/markdown content (basic cleaning)
        content = re.sub(r"<script.*?</script>", "", content, flags=re.DOTALL)
        content = re.sub(r"<style.*?</style>", "", content, flags=re.DOTALL)
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).strip()

        # Split into chunks if too large
        documents = []
        if len(content) <= chunk_size:
            doc = Document(
                text=content,
                metadata={
                    "title": title,
                    "source": "url",
                    "url": url,
                },
            )
            documents.append(doc)
        else:
            # Split into multiple documents
            num_chunks = (len(content) + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(content))
                chunk = content[start:end]

                doc = Document(
                    text=chunk,
                    metadata={
                        "title": f"{title} (Part {i + 1}/{num_chunks})",
                        "source": "url",
                        "url": url,
                    },
                )
                documents.append(doc)

        return documents

    def ingest_dependency_docs(
        self,
        package_name: str,
        version: str | None = None,
        github_repo: tuple[str, str] | None = None,
        doc_urls: list[tuple[str, str]] | None = None,
    ) -> dict[str, list[Document]]:
        """Ingest all available documentation for a dependency.

        Args:
            package_name: Name of the package
            version: Specific version, or None for latest
            github_repo: Optional tuple of (owner, repo_name)
            doc_urls: Optional list of (url, title) tuples for official docs

        Returns:
            Dictionary mapping source type to list of documents
        """
        results = {}

        # Fetch from PyPI
        print(f"Fetching PyPI documentation for {package_name}...")
        pypi_docs = self.create_documents_from_pypi(package_name, version)
        if pypi_docs:
            results["pypi"] = pypi_docs
            # Get actual version from PyPI
            _, actual_version = self.fetch_pypi_info(package_name, version)
            version = actual_version

        time.sleep(0.5)  # Rate limiting

        # Fetch from GitHub if repo provided
        if github_repo and version:
            owner, repo = github_repo
            tag = f"v{version}" if not version.startswith("v") else version
            print(f"Fetching GitHub release notes for {owner}/{repo} {tag}...")
            github_docs = self.create_documents_from_github_release(owner, repo, tag, package_name)
            if github_docs:
                results["github"] = github_docs
            time.sleep(0.5)

        # Fetch from documentation URLs if provided
        if doc_urls:
            for url, title in doc_urls:
                print(f"Fetching documentation from {url}...")
                url_docs = self.create_documents_from_url(url, title, package_name)
                if url_docs:
                    source_key = f"url_{len(results)}"
                    results[source_key] = url_docs
                time.sleep(1.0)  # Longer delay for external docs

        return results
