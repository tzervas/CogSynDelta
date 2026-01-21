# License Tracker

This document tracks all third-party dependencies, their licenses, and compliance requirements for CogSynDelta.

**CogSynDelta License**: Proprietary
**Copyright**: © 2026 Tyler Zervas (tzervas) and Average Joe's Labs (AJL)

## License Compatibility Matrix

| License | Compatible with Proprietary | Commercial Use | Requires Attribution | Copyleft |
|---------|---------------------|----------------|----------------------|----------|
| MIT | ✅ | ✅ | ✅ | ❌ |
| Apache-2.0 | ✅ | ✅ | ✅ | ❌ |
| BSD-2-Clause | ✅ | ✅ | ✅ | ❌ |
| BSD-3-Clause | ✅ | ✅ | ✅ | ❌ |
| PSF-2.0 | ✅ | ✅ | ✅ | ❌ |
| ISC | ✅ | ✅ | ✅ | ❌ |
| LGPL-2.1+ | ⚠️ | ✅ | ✅ | ⚠️ (weak) |
| GPL-3.0 | ❌ | ⚠️ | ✅ | ✅ (strong) |

**Legend**: ✅ Yes | ❌ No | ⚠️ Conditional

## Core Dependencies

### Machine Learning

| Package | Version | License | Usage | Compliance Notes |
|---------|---------|---------|-------|------------------|
| torch | >=2.9.0 | BSD-3-Clause | Core neural networks | Attribution in docs |
| torchvision | >=0.24.0 | BSD-3-Clause | Image processing | Attribution in docs |
| numpy | >=2.4.0 | BSD-3-Clause | Numerical operations | Attribution in docs |

### API & Web

| Package | Version | License | Usage | Compliance Notes |
|---------|---------|---------|-------|------------------|
| fastapi | >=0.128.0 | MIT | REST API server | None required |
| uvicorn | >=0.40.0 | BSD-3-Clause | ASGI server | Attribution in docs |
| pydantic | >=2.12.0 | MIT | Data validation | None required |
| websockets | >=16.0 | BSD-3-Clause | WebSocket support | Attribution in docs |

### Memory & Search

| Package | Version | License | Usage | Compliance Notes |
|---------|---------|---------|-------|------------------|
| llama-index-core | >=0.14.0 | MIT | RAG pipeline | None required |
| llama-index-embeddings-huggingface | >=0.6.0 | MIT | Embeddings | None required |
| llama-index-vector-stores-faiss | >=0.5.0 | MIT | Vector search | None required |
| faiss-cpu | >=1.9.0 | MIT | Similarity search | None required |

### HTTP & Async

| Package | Version | License | Usage | Compliance Notes |
|---------|---------|---------|-------|------------------|
| httpx | >=0.28.0 | BSD-3-Clause | HTTP client | Attribution in docs |
| aiohttp | >=3.11.0 | Apache-2.0 | Async HTTP | Include NOTICE file |
| python-multipart | >=0.0.21 | Apache-2.0 | Form handling | Include NOTICE file |

### Configuration

| Package | Version | License | Usage | Compliance Notes |
|---------|---------|---------|-------|------------------|
| pyyaml | >=6.0.0 | MIT | YAML config | None required |

## Optional Dependencies

### Vision (`[vision]`)

| Package | Version | License | Usage | Compliance Notes |
|---------|---------|---------|-------|------------------|
| opencv-python | >=4.10.0 | Apache-2.0 | Image processing | Include NOTICE |
| mss | >=10.0.0 | MIT | Screen capture | None required |
| pillow | >=11.0.0 | HPND | Image handling | Attribution in docs |

### GPU Acceleration (`[gpu-nvidia]`)

| Package | Version | License | Usage | Compliance Notes |
|---------|---------|---------|-------|------------------|
| triton | >=3.3.0 | MIT | Kernel optimization | None required |

### Inference Providers (`[inference-providers]`)

| Package | Version | License | Usage | Compliance Notes |
|---------|---------|---------|-------|------------------|
| openai | >=2.15.0 | MIT | OpenAI API | None required |
| anthropic | >=0.76.0 | MIT | Claude API | None required |
| google-cloud-aiplatform | >=1.86.0 | Apache-2.0 | Vertex AI | Include NOTICE |

## Development Dependencies

| Package | Version | License | Usage | Compliance Notes |
|---------|---------|---------|-------|------------------|
| pytest | >=9.0.0 | MIT | Testing | Dev only |
| pytest-cov | >=7.0.0 | MIT | Coverage | Dev only |
| pytest-asyncio | >=1.3.0 | Apache-2.0 | Async tests | Dev only |
| ruff | >=0.14.13 | MIT | Linting | Dev only |
| mypy | >=1.19.0 | MIT | Type checking | Dev only |
| pre-commit | >=4.5.0 | MIT | Git hooks | Dev only |

## Blocked Dependencies

These packages are NOT included due to license or compatibility issues:

| Package | Issue | Alternative |
|---------|-------|-------------|
| qiskit | No Python 3.14 wheels | Classical simulation stubs |
| pennylane | No Python 3.14 wheels | Classical simulation stubs |
| cirq | typedunits dependency issue | Classical simulation stubs |
| fireworks-ai | ruff version conflict | Other inference providers |

## License Files

Full license texts for attributed packages are in `LICENSES/`:

- `MIT.txt` - MIT License template
- `Apache-2.0.txt` - Apache 2.0 License
- `BSD-3-Clause.txt` - BSD 3-Clause License

## Compliance Checklist

### For Releases

- [ ] All dependencies have compatible licenses
- [ ] Attribution requirements met in documentation
- [ ] NOTICE files included for Apache-2.0 dependencies
- [ ] No GPL/LGPL dependencies without explicit approval

### For New Dependencies

Before adding a new dependency:

1. Check license compatibility with MIT
2. Add to this tracker with usage notes
3. Update `LICENSES/` if new license type
4. Run `scripts/license_audit.py` to verify

## Audit Script

Run the license audit:

```bash
uv run python scripts/license_audit.py
```

This script:
1. Scans installed packages
2. Compares against this tracker
3. Flags unknown or incompatible licenses
4. Generates compliance report

## Future Licensing Considerations

### Open Source Release

If releasing as open source:
- MIT license is compatible with all current dependencies
- Need to include attribution notices
- Apache-2.0 NOTICE files must be preserved

### Proprietary Release

If releasing as proprietary:
- All current dependencies allow commercial use
- Need to maintain attribution in documentation
- Consider replacing LGPL dependencies if any added

---

**Last Updated**: 2026-01-18
**Audited By**: @tzervas
