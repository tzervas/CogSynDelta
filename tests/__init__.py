"""CogSynDelta test package.

Why: GPU runner images ship a site-packages ``tests`` module. A real
package here (``__init__.py``) makes ``import tests.test_poc_cuda``
resolve to this tree when the repo root is first on ``sys.path``.
"""
